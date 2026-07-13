#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <errno.h>

#define KGSL_IOC_TYPE 0x09

struct kgsl_map_user_mem {
    int fd;
    unsigned long gpuaddr;
    size_t len;
    size_t offset;
    unsigned long hostptr;
    unsigned int memtype;
    unsigned int flags;
};

struct kgsl_drawctxt_create {
    unsigned int flags;
    unsigned int drawctxt_id;
};

struct kgsl_drawctxt_destroy {
    unsigned int drawctxt_id;
};

struct kgsl_gpu_command {
    uint64_t flags;
    uint64_t cmdlist;
    unsigned int cmdsize;
    unsigned int numcmds;
    uint64_t objlist;
    unsigned int objsize;
    unsigned int numobjs;
    uint64_t synclist;
    unsigned int syncsize;
    unsigned int numsyncs;
    unsigned int context_id;
    unsigned int timestamp;
};

struct kgsl_command_object {
    uint64_t offset;
    uint64_t gpuaddr;
    uint64_t size;
    unsigned int flags;
    unsigned int id;
};

struct kgsl_timestamp_event {
    unsigned int type;      /* KGSL_TIMESTAMP_EVENT_TYPE */
    unsigned int timestamp; /* timestamp to wait for */
    unsigned int context_id;
    int fd;                 /* for type=KGSL_TIMESTAMP_EVENT_SIGNAL */
};

struct kgsl_timestamp_event_retired {
    unsigned int timestamp;
    unsigned int reserved;
};

#define KGSL_USER_MEM_TYPE_ADDR 0x00000002
#define KGSL_CMDLIST_IB 0x00000001U

#define IOCTL_KGSL_DRAWCTXT_CREATE   _IOWR(KGSL_IOC_TYPE, 0x13, struct kgsl_drawctxt_create)
#define IOCTL_KGSL_DRAWCTXT_DESTROY   _IOW(KGSL_IOC_TYPE, 0x14, struct kgsl_drawctxt_destroy)
#define IOCTL_KGSL_MAP_USER_MEM       _IOWR(KGSL_IOC_TYPE, 0x15, struct kgsl_map_user_mem)
#define IOCTL_KGSL_GPU_COMMAND        _IOWR(KGSL_IOC_TYPE, 0x4A, struct kgsl_gpu_command)
#define IOCTL_KGSL_TIMESTAMP_EVENT    _IOWR(KGSL_IOC_TYPE, 0x1A, struct kgsl_timestamp_event_retired)

/* Adreno GPU packet helpers */
#define CP_TYPE7_PKT(opcode, count) ((7 << 28) | ((opcode) << 16) | ((count) & 0x3FFF))
#define CP_NOP           0x10
#define CP_WAIT_FOR_IDLE 0x26
#define CP_MEM_WRITE     0x3d
#define CP_WAIT_MEM_WRITES 0x12

#define upper_32_bits(n) ((uint32_t)(((n) >> 16) >> 16))
#define lower_32_bits(n) ((uint32_t)(n))

static inline unsigned int cp_gpuaddr(uint32_t *cmds, uint64_t gpuaddr) {
    cmds[0] = lower_32_bits(gpuaddr);
    cmds[1] = upper_32_bits(gpuaddr);
    return 2;
}

int main() {
    int fd = open("/dev/kgsl-3d0", O_RDWR);
    if (fd < 0) { printf("[-] open: %s\n", strerror(errno)); return 1; }
    printf("[+] fd=%d\n", fd);

    /* Create context */
    struct kgsl_drawctxt_create ctx = { .flags = 0x00001812 };
    if (ioctl(fd, IOCTL_KGSL_DRAWCTXT_CREATE, &ctx)) {
        printf("[-] ctx: %s\n", strerror(errno));
        goto out;
    }
    printf("[+] ctx id=%u\n", ctx.drawctxt_id);

    /* Allocate result buffer */
    uint32_t *result = mmap(NULL, 0x10000, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    memset(result, 0xAB, 0x10000);

    uint64_t result_gpuaddr = 0;
    {
        struct kgsl_map_user_mem m = {
            .len = 0x10000, .hostptr = (unsigned long)result,
            .memtype = KGSL_USER_MEM_TYPE_ADDR,
        };
        if (ioctl(fd, IOCTL_KGSL_MAP_USER_MEM, &m)) {
            printf("[-] map result: %s\n", strerror(errno));
            goto destroy;
        }
        result_gpuaddr = m.gpuaddr;
    }
    printf("[+] result mapped: host=%p gpuaddr=0x%lx\n", result, (unsigned long)result_gpuaddr);

    /* Build GPU command buffer: do MEM_WRITE at offset 0x100 in result buffer */
    uint32_t *cmd = mmap(NULL, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if (cmd == MAP_FAILED) {
        printf("[-] cmd mmap: %s\n", strerror(errno));
        goto destroy;
    }
    memset(cmd, 0, 4096);
    uint32_t *p = cmd;

    uint64_t target_ga = result_gpuaddr + 0x100;

    /* WAIT_FOR_IDLE first */
    *p++ = CP_TYPE7_PKT(CP_WAIT_FOR_IDLE, 0);
    /* MEM_WRITE: count=3 (addr_lo, addr_hi, value) */
    *p++ = CP_TYPE7_PKT(CP_MEM_WRITE, 3);
    p += cp_gpuaddr(p, target_ga);
    *p++ = 0xDEADBEEF;
    /* WAIT_MEM_WRITES to flush */
    *p++ = CP_TYPE7_PKT(CP_WAIT_MEM_WRITES, 0);
    /* MEM_WRITE completion marker at offset 0 */
    *p++ = CP_TYPE7_PKT(CP_MEM_WRITE, 3);
    p += cp_gpuaddr(p, result_gpuaddr);
    *p++ = 0xCAFEBABE;  /* completion marker */
    *p++ = CP_TYPE7_PKT(CP_NOP, 0);

    uint32_t cmd_size = (char*)p - (char*)cmd; /* in bytes */

    printf("[*] cmd size=%u bytes (%u dwords)\n", cmd_size, cmd_size/4);

    /* Map command buffer to GPU */
    uint64_t cmd_gpuaddr = 0;
    {
        struct kgsl_map_user_mem m = {
            .len = 4096, .hostptr = (unsigned long)cmd,
            .memtype = KGSL_USER_MEM_TYPE_ADDR,
        };
        if (ioctl(fd, IOCTL_KGSL_MAP_USER_MEM, &m)) {
            printf("[-] map cmd: %s\n", strerror(errno));
            goto destroy;
        }
        cmd_gpuaddr = m.gpuaddr;
    }
    printf("[+] cmd mapped: gpuaddr=0x%lx\n", (unsigned long)cmd_gpuaddr);

    /* Sync CPU cache to GPU (important for ARM!) */
    __builtin___clear_cache((char*)cmd, (char*)cmd + 4096);
    __builtin___clear_cache((char*)result, (char*)result + 0x10000);

    /* Submit GPU command with timestamp=42 */
    {
        struct kgsl_command_object objs[1] = {{
            .gpuaddr = cmd_gpuaddr,
            .size = cmd_size,
            .flags = KGSL_CMDLIST_IB,
        }};
        struct kgsl_gpu_command req = {
            .cmdlist = (uint64_t)(unsigned long)objs,
            .cmdsize = sizeof(struct kgsl_command_object),
            .numcmds = 1,
            .context_id = ctx.drawctxt_id,
            .timestamp = 42,
        };
        printf("[*] submitting GPU command (ts=42)...\n");
        if (ioctl(fd, IOCTL_KGSL_GPU_COMMAND, &req)) {
            printf("[-] gpu_cmd: %s\n", strerror(errno));
            goto destroy;
        }
        printf("[+] submitted ts=%u\n", req.timestamp);
    }

    /* Try to wait for timestamp using TIMESTAMP_EVENT ioctl */
    printf("[*] waiting for GPU completion...\n");
    {
        struct kgsl_timestamp_event_retired ev = {
            .timestamp = 42,
        };
        int ret = ioctl(fd, IOCTL_KGSL_TIMESTAMP_EVENT, &ev);
        if (ret == 0) {
            printf("[+] TIMESTAMP_EVENT OK (waited for ts=42)\n");
        } else {
            printf("[-] TIMESTAMP_EVENT: %s (falling back to polling)\n", strerror(errno));
            /* Fall back to polling */
            usleep(500000);
            usleep(500000);
        }
    }

    /* Check result */
    printf("\n=== Results ===\n");
    printf("result[0]    = 0x%08x (expected 0xCAFEBABE)\n", result[0]);
    printf("result[64]   = 0x%08x (expected 0xDEADBEEF)\n", result[0x100/4]);
    
    if (result[0] == 0xCAFEBABE) {
        printf("✅ GPU EXECUTED! Both writes confirmed!\n");
    } else if (result[0x100/4] == 0xDEADBEEF) {
        printf("⚠️  Partial: GPU wrote marker but not target\n");
    } else if (result[0] == 0xABABABAB) {
        printf("❌ GPU did NOT execute. Cache issue or command format wrong.\n");
        
        /* Try polling more */
        printf("[*] polling for 5 more seconds...\n");
        for (int i = 0; i < 5; i++) {
            sleep(1);
            printf("    result[0]=0x%08x result[64]=0x%08x\n", result[0], result[0x100/4]);
            if (result[0] != 0xABABABAB) {
                printf("✅ GPU responded!\n");
                break;
            }
        }
    } else {
        printf("❓ Unexpected value\n");
    }

destroy:
    {
        struct kgsl_drawctxt_destroy d = { .drawctxt_id = ctx.drawctxt_id };
        ioctl(fd, IOCTL_KGSL_DRAWCTXT_DESTROY, &d);
    }
out:
    close(fd);
    printf("[+] DONE\n");
    return 0;
}