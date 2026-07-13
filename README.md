# watch4pro-root

Huawei Watch 4 Pro (MDS-AL00) root exploit research.

**Device:** Huawei Watch 4 Pro (MDS-AL00)  
**SoC:** Snapdragon W5 Gen 1 (SW5100)  
**GPU:** Adreno 702  
**Kernel:** 5.4.161-perf (armv7l)  
**OS:** HarmonyOS 12  

## Approach

Leveraging Qualcomm KGSL (Graphics System Layer) driver vulnerabilities to gain arbitrary kernel memory read/write via the GPU's SMMU, then disable SELinux and escalate to root.

## Build

```bash
NDK=/opt/android-sdk/ndk/25.2.9519653
TOOLCHAIN=$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin
$TOOLCHAIN/armv7a-linux-androideabi21-clang -marm -static -O2 src/foo.c -o binaries/foo
```

## Progress

- [x] KGSL device accessible (/dev/kgsl-3d0, mode 666)
- [ ] KGSL driver version identified
- [ ] GPU context creation confirmed
- [ ] User memory mapping to GPU
- [ ] GPU command submission
- [ ] SMMU page table manipulation
- [ ] Kernel physical memory R/W
- [ ] SELinux bypass
- [ ] Root shell
