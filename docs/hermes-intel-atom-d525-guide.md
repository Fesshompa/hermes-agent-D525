# Running Hermes Agent on an Intel Atom D525

## Summary

We got Hermes Agent running on a 64-bit Debian system powered by an Intel Atom D525 by fixing a CPU-instruction incompatibility in NumPy, then ensuring Hermes kept that compatible NumPy build whenever it rebuilt its managed Python environment.

This is a compatibility workaround, not a way to make the Atom fast. Hermes can run its agent, tools and messaging integrations on the D525, but model inference will generally need a remote API or another machine. This guide describes the path tested on Debian with a Hermes source installation using CPython 3.14; paths and internal package-manager code differ between installations and Hermes releases.

## 1. Understand the failure

The D525 supports SSE2, SSE3 and SSSE3, but not SSE4.1 or SSE4.2. Check the actual CPU flags on the machine:

```bash
lscpu
 grep -m1 '^flags' /proc/cpuinfo
```

On this CPU, the `flags` line should not contain `sse4_1` or `sse4_2`. A NumPy native extension compiled for an instruction the processor does not support can terminate Python with `SIGILL` (signal 4, “illegal instruction”) as soon as that extension is imported or an affected code path runs.

The stock NumPy wheel we tested crashed on the D525. The failing native code contained an SSE4.1 instruction. The solution was to build NumPy 2.4.3 locally with newer CPU instruction baselines and dispatch paths disabled.

NumPy's build documentation explains why this matters: the default x86-64 baseline can be too new for pre-2009 CPUs; use `cpu-baseline=none` for old x86 CPUs, and also set `cpu-dispatch=none` if you want to avoid building newer dispatched code paths. Setting only the baseline does **not** disable dispatch. See the [NumPy CPU build options](https://numpy.org/devdocs/reference/simd/build-options.html).

## 2. Build a D525-compatible NumPy wheel

Build for the same operating system architecture and Python minor version as Hermes uses. Our tested Hermes environment used **CPython 3.14**, so its wheel was named `numpy-2.4.3-cp314-cp314-linux_x86_64.whl`. If your Hermes installation uses another Python version, build the wheel with that version instead.

### Install build tools

On Debian, install a compiler toolchain and ensure the matching Python interpreter and headers are available. For example:

```bash
apt-get update
apt-get install -y build-essential gfortran pkg-config
python3.14 --version
```

Use the Python 3.14 interpreter supplied by your Hermes installation if `python3.14` is not the same version Hermes uses. On our host, Hermes-managed Python was under `/var/lib/hermes/.hermes/tools/`.

Create an isolated build environment and download the NumPy source release:

```bash
python3.14 -m venv /tmp/numpy-build-venv
. /tmp/numpy-build-venv/bin/activate
python -m pip install --upgrade pip build
python -m pip download --no-deps --no-binary=:all: numpy==2.4.3 -d /tmp/numpy-src
tar -xzf /tmp/numpy-src/numpy-2.4.3.tar.gz -C /tmp/numpy-src
mkdir -p /opt/hermes-local-wheels
```

Build on the D525 itself where practical. Clear compiler environment flags so they cannot silently inject CPU-specific options such as `-march=native`, then build the wheel:

```bash
unset CFLAGS CXXFLAGS CPPFLAGS
cd /tmp/numpy-src/numpy-2.4.3
python -m build --wheel --outdir /opt/hermes-local-wheels \
  -Csetup-args=-Dcpu-baseline=none \
  -Csetup-args=-Dcpu-dispatch=none \
  -Csetup-args=-Dcpu-baseline-detect=disabled
```

The build may take a long time on this CPU. Do not interrupt it just because compilation is slow. Confirm the output filename and record its SHA-256:

```bash
ls -lh /opt/hermes-local-wheels/numpy-2.4.3-*.whl
sha256sum /opt/hermes-local-wheels/numpy-2.4.3-*.whl
```

The hash of the wheel built during our successful run was:

```text
57aa4919e213903947f9bb3d7474566705226117a6ce3b8eeaed8d684b87f907  numpy-2.4.3-cp314-cp314-linux_x86_64.whl
```

That hash is only valid for that exact wheel file. If you build your own wheel, use the hash you calculate locally; do not copy this hash to validate a different file.

### Test the wheel before changing Hermes

Use a disposable environment built with the same Python version. If your build Python's venv includes pip, run:

```bash
python -m venv /tmp/numpy-atom-test
/tmp/numpy-atom-test/bin/python -m pip install --no-index \
  --find-links /opt/hermes-local-wheels numpy==2.4.3
/tmp/numpy-atom-test/bin/python - <<'PY'
import numpy as np
print("NumPy:", np.__version__)
print("Dot product:", int(np.dot(np.arange(100), np.arange(100))))
print("Solve:", np.linalg.solve(
    np.array([[2., 1.], [1., 3.]]), np.array([7., 11.])
).tolist())
PY
```

Expected output includes NumPy `2.4.3`, dot product `328350`, and solve result `[2.0, 3.0]`. Most importantly, the process must not crash with `Illegal instruction`. If the wheel is for the wrong Python ABI or architecture, the install will not accept it.

## 3. Make Hermes apply the wheel every time it creates an environment

**Do not only install this wheel into the current Hermes virtual environment.** Hermes owns and may recreate that environment from its lockfile; a later repair or update can reinstall the incompatible public NumPy wheel. Hermes' package-management documentation describes its managed, versioned environments and selection metadata: [Hermes package management](https://github.com/NousResearch/hermes-agent/blob/main/website/docs/reference/package-management.md).

In our source installation, we made a small local change to:

```text
/usr/local/lib/hermes-agent/pm/packages.py
```

The change added an `_install_atom_compatible_numpy(environment)` hook **after the regular locked dependency installation and before the environment validation/check**. The hook does the following:

1. Runs only on Linux x86-64 systems where `/proc/cpuinfo` indicates that `sse4_1` is absent.
2. Checks that the local wheel exists and its SHA-256 matches a configured, trusted value.
3. Checks that the wheel's Python version matches the environment (CPython 3.14 in our case).
4. Uses Hermes' environment/runner abstraction to reinstall `numpy==2.4.3` from `/opt/hermes-local-wheels`, with package-index access disabled, binary wheels only, and `--reinstall` enabled.
5. Lets Hermes validate the environment after the replacement finishes.

The call sequence is conceptually:

```python
lock_and_sync(...)
_install_atom_compatible_numpy(environment)
environment.check()
```

This is an internal, version-specific extension point, not a supported public Hermes plugin API. Back up `packages.py` first and inspect the surrounding code in **your installed release** before adapting it. The environment object's Python/runner interface may differ across releases; use the interface already used by that version rather than copying guessed attribute names from another release.

A safe hook must fail closed: if the wheel is missing, has the wrong hash, has the wrong Python ABI, or cannot be installed, the environment build should fail rather than quietly continue with the incompatible NumPy wheel. Keep the local wheel root-owned but readable by the Hermes service account (typically directory mode `755`, wheel mode `644`). Keep the source patch under change control, as an Hermes update may replace it.

After adding the hook, trigger a **fresh Hermes-managed environment build/repair** using the supported PM command for your release (for releases that provide it, `hermes pm repair`). Preserve the installation's existing optional-feature/plugin set when rebuilding; don't copy another host's extras blindly. Perform the PM operation as the service account where possible so that the selected-environment metadata and candidate files remain owned and readable by Hermes. Official PM docs recommend using Hermes PM operations rather than ad-hoc edits to managed environments.

## 4. Verify the selected Hermes environment

Find the newly selected environment with your Hermes PM status/diagnostic commands, or inspect the installation's per-install `facts.json` if you understand that release's metadata format. Do not guess the candidate directory. With the selected Python path in `PY`, test as the service account:

```bash
su -s /bin/sh hermes -c "'$PY' -c 'import numpy as np; print(np.__version__, int(np.dot(np.arange(100), np.arange(100))))'"
su -s /bin/sh hermes -c "'$PY' -m hermes_cli.main --help >/dev/null"
```

The first command should print `2.4.3 328350`; the second should exit successfully. Hermes' managed environment may not contain the `pip` module, so `python -m pip check` can fail with `No module named pip` even when the environment is healthy. Use Hermes' supported PM diagnostics or its bundled `uv pip check --python "$PY"` if `uv` is available.

### Ownership pitfall we hit

We initially generated the candidate as `root`, which left root-owned selection metadata that the `hermes` service account could not read. The first CLI launch then failed with a permission error and attempted a repair. We corrected ownership of the candidate and `facts.json`, then allowed Hermes to complete repair and verified NumPy again.

Avoid this by doing the PM build/repair as `hermes`. If repair fails with `Permission denied` for `facts.json`, stop and inspect ownership and permissions; do not delete markers, hand-edit selection metadata, or recursively `chown` the entire Hermes home as a reflex. Restrict any ownership correction to the exact candidate and selection files after confirming paths and the service account.

## 5. Point systemd at the selected environment

On our machine, the systemd units hard-coded the old interpreter under `/usr/local/lib/hermes-agent/venv/`. We added drop-ins for `hermes-gateway.service` and `hermes-dashboard.service` to use the selected candidate's Python executable instead.

First inspect the real units and existing drop-ins:

```bash
systemctl cat hermes-gateway.service hermes-dashboard.service
systemctl show hermes-gateway.service -p ExecStart -p Environment
```

Create a `candidate-env.conf` drop-in in each service's existing `.d/` directory. In each drop-in, clear and replace `ExecStart` with the selected environment's Python command, and set `VIRTUAL_ENV` plus a `PATH` that starts with the selected environment's `bin` directory. For example, the gateway command was:

```text
/var/lib/hermes/.hermes/installs/76f6e7e5145b5f39/environments/9c287b18aa24498abdc05bc914324c3d/venv/bin/python -m hermes_cli.main gateway run
```

The dashboard command was the same Python followed by:

```text
-m hermes_cli.main dashboard --no-open
```

Those are example paths from our installation. Use your own selected environment path, and preserve the other required settings from your actual units. In particular, do **not** remove the gateway's existing `memory.conf` or other environment settings when adding a drop-in.

After writing the drop-ins:

```bash
systemctl daemon-reload
systemctl restart hermes-gateway.service hermes-dashboard.service
SYSTEMD_PAGER=cat systemctl show hermes-gateway.service hermes-dashboard.service \
  -p Id -p ActiveState -p SubState -p MainPID -p ExecStart
journalctl -u hermes-gateway.service -n 40 --no-pager
journalctl -u hermes-dashboard.service -n 25 --no-pager
```

Restart only the Hermes services you are changing. Do not restart unrelated providers or dependencies unnecessarily.

## 6. Final acceptance checks

Confirm all of the following:

- The active gateway and dashboard `ExecStart` paths point to the new environment.
- The gateway remains `active/running` and its restart count does not keep increasing.
- The gateway logs show normal startup and messaging integration connection (Telegram connected in our test).
- The dashboard log says `HERMES_DASHBOARD_READY` and names its port (9119 in our installation).
- NumPy import and numerical tests pass under the Hermes service account.
- Memory use remains under the configured service limit. Our gateway settled at about 292 MB current memory with a 600 MiB maximum.

## What this fixes—and what it does not

This fixes the NumPy CPU-instruction incompatibility we encountered. It does not guarantee every future native Python dependency will support the D525; if another package crashes with `SIGILL`, inspect that library separately. It also does not provide practical local LLM inference performance. The D525 can host the Hermes agent process, but model calls should use a reachable remote inference service or another machine.

## References

- NumPy, [CPU build options](https://numpy.org/devdocs/reference/simd/build-options.html): explains `cpu-baseline` and `cpu-dispatch`, including the need to disable dispatch for broad old-x86 compatibility.
- Intel, [Instruction Set Extensions Technology](https://www.intel.com/content/www/us/en/support/articles/000005779/processors.html): overview of the SSE instruction families.
- Hermes Agent, [Package Management](https://github.com/NousResearch/hermes-agent/blob/main/website/docs/reference/package-management.md): managed Python environments, selection metadata and supported PM operations.
