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

#define KGSL_USER_MEM_TYPE_ADDR 0x00000002
#define KGSL_CMDLIST_IB 0x00000001U

#define IOCTL_KGSL_DRAWCTXT_CREATE   _IOWR(KGSL_IOC_TYPE, 0x13, struct kgsl_drawctxt_create)
#define IOCTL_KGSL_DRAWCTXT_DESTROY   _IOW(KGSL_IOC_TYPE, 0x14, struct kgsl_drawctxt_destroy)
#define IOCTL_KGSL_MAP_USER_MEM       _IOWR(KGSL_IOC_TYPE, 0x15, struct kgsl_map_user_mem)
#define IOCTL_KGSL_GPU_COMMAND        _IOWR(KGSL_IOC_TYPE, 0x4A, struct kgsl_gpu_command)

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
        goto out1;
    }
    printf("[+] ctx id=%u\n", ctx.drawctxt_id);

    /* == Allocate a single large buffer: commands at offset 0, result at offset 0x1000 == */
    uint32_t *buf = mmap(NULL, 0x8000, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if (buf == MAP_FAILED) {
        printf("[-] mmap buf: %s\n", strerror(errno));
        goto out2;
    }
    memset(buf, 0xAB, 0x8000);
    
    /* Command buffer at offset 0 */
    uint32_t *cmd = buf;
    uint32_t *p = cmd;
    
    uint64_t buf_gpuaddr = 0;
    uint64_t cmd_gpuaddr = 0;  /* same as buf_gpuaddr */
    uint64_t result_gpuaddr = 0; /* buf_gpuaddr + 0x1000 */
    
    /* Build GPU commands */
    /* WAIT_FOR_IDLE */
    *p++ = CP_TYPE7_PKT(CP_WAIT_FOR_IDLE, 0);
    /* MEM_WRITE to result_gpuaddr + 0x000 (completion marker) */
    *p++ = CP_TYPE7_PKT(CP_MEM_WRITE, 3);
    /* Address will be filled after mapping */
    uint32_t *completion_addr_lo = p;
    uint32_t *completion_addr_hi = p + 1;
    p += 2;
    *p++ = 0xCAFEBABE;  /* completion marker */
    /* MEM_WRITE to result_gpuaddr + 0x100 */
    *p++ = CP_TYPE7_PKT(CP_MEM_WRITE, 3); 
    uint32_t *target_addr_lo = p;
    uint32_t *target_addr_hi = p + 1;
    p += 2;
    *p++ = 0xDEADBEEF;  /* target value */
    /* WAIT_MEM_WRITES */
    *p++ = CP_TYPE7_PKT(CP_WAIT_MEM_WRITES, 0);
    *p++ = CP_TYPE7_PKT(CP_NOP, 0);
    
    uint32_t cmd_size = (char*)p - (char*)cmd;
    printf("[*] cmd size=%u bytes\n", cmd_size);
    
    /* Map the entire buffer to GPU */
    {
        struct kgsl_map_user_mem m = {
            .len = 0x8000,
            .hostptr = (unsigned long)buf,
            .memtype = KGSL_USER_MEM_TYPE_ADDR,
        };
        int ret = ioctl(fd, IOCTL_KGSL_MAP_USER_MEM, &m);
        if (ret) {
            printf("[-] map buf: %s (errno=%d)\n", strerror(errno), errno);
            goto out2;
        }
        buf_gpuaddr = m.gpuaddr;
        cmd_gpuaddr = buf_gpuaddr;
        result_gpuaddr = buf_gpuaddr + 0x1000;
    }
    printf("[+] buf mapped: host=%p gpuaddr=0x%lx\n", buf, (unsigned long)buf_gpuaddr);
    printf("[+] cmd at 0x%lx, result at 0x%lx\n", (unsigned long)cmd_gpuaddr, (unsigned long)result_gpuaddr);
    
    /* Fill in the GPU addresses in the command buffer */
    *completion_addr_lo = lower_32_bits(result_gpuaddr);
    *completion_addr_hi = upper_32_bits(result_gpuaddr);
    *target_addr_lo = lower_32_bits(result_gpuaddr + 0x100);
    *target_addr_hi = upper_32_bits(result_gpuaddr + 0x100);
    
    /* Sync cache */
    __builtin___clear_cache((char*)buf, (char*)buf + 0x8000);
    
    /* Submit GPU command */
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
        printf("[*] submitting GPU command...\n");
        int ret = ioctl(fd, IOCTL_KGSL_GPU_COMMAND, &req);
        if (ret) {
            printf("[-] gpu_cmd: %s (errno=%d)\n", strerror(errno), errno);
            goto out2;
        }
        printf("[+] submitted ts=%u\n", req.timestamp);
    }
    
    /* Poll for completion */
    printf("[*] polling for completion...\n");
    volatile uint32_t *result = (volatile uint32_t *)((char*)buf + 0x1000);
    for (int i = 0; i < 20; i++) {
        uint32_t marker = result[0];
        uint32_t target = result[0x100/4];
        printf("    [%d] marker=0x%08x target=0x%08x\n", i, marker, target);
        if (marker == 0xCAFEBABE) {
            printf("✅ GPU EXECUTED! Marker=0xCAFEBABE, target=0x%08x\n", target);
            if (target == 0xDEADBEEF) {
                printf("✅ Both writes confirmed! GPU can write to mapped memory!\n");
            } else {
                printf("⚠️  Marker OK but target=0x%08x (expected 0xDEADBEEF)\n", target);
            }
            goto out2;
        }
        usleep(100000);
    }
    
    printf("❌ GPU did not respond after 2 seconds\n");
    printf("   result[0]=0x%08x result[0x100/4]=0x%08x\n", result[0], result[0x100/4]);

out2:
    {
        struct kgsl_drawctxt_destroy d = { .drawctxt_id = ctx.drawctxt_id };
        ioctl(fd, IOCTL_KGSL_DRAWCTXT_DESTROY, &d);
    }
out1:
    close(fd);
    printf("[+] DONE\n");
    return 0;
}