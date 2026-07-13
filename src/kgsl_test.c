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

struct kgsl_version {
    unsigned int drv_major;
    unsigned int drv_minor;
    unsigned int dev_major;
    unsigned int dev_minor;
};

#define IOCTL_KGSL_DRAWCTXT_CREATE _IOWR(KGSL_IOC_TYPE, 0x13, struct kgsl_drawctxt_create)
#define IOCTL_KGSL_DRAWCTXT_DESTROY _IOW(KGSL_IOC_TYPE, 0x14, struct kgsl_drawctxt_destroy)
#define IOCTL_KGSL_MAP_USER_MEM _IOWR(KGSL_IOC_TYPE, 0x15, struct kgsl_map_user_mem)
#define IOCTL_KGSL_GPU_COMMAND _IOWR(KGSL_IOC_TYPE, 0x4A, struct kgsl_gpu_command)
#define IOCTL_KGSL_DEVICE_GET_VERSION _IOWR(KGSL_IOC_TYPE, 0x09, struct kgsl_version)

#define KGSL_CMDLIST_IB 0x00000001U
#define KGSL_USER_MEM_TYPE_ADDR 0x00000002

int main() {
    int fd = open("/dev/kgsl-3d0", O_RDWR);
    if (fd < 0) {
        printf("[-] open /dev/kgsl-3d0: %s (errno=%d)\n", strerror(errno), errno);
        return 1;
    }
    printf("[+] fd=%d\n", fd);

    /* 1. Get KGSL version */
    struct kgsl_version ver;
    memset(&ver, 0, sizeof(ver));
    int ret = ioctl(fd, IOCTL_KGSL_DEVICE_GET_VERSION, &ver);
    if (ret < 0) {
        printf("[-] GET_VERSION: %s\n", strerror(errno));
    } else {
        printf("[+] KGSL: drv=%u.%u dev=%u.%u\n",
               ver.drv_major, ver.drv_minor,
               ver.dev_major, ver.dev_minor);
    }

    /* 2. Try create context */
    struct kgsl_drawctxt_create ctx = { .flags = 0x00001812 };
    ret = ioctl(fd, IOCTL_KGSL_DRAWCTXT_CREATE, &ctx);
    if (ret < 0) {
        printf("[-] CREATE CTX: %s\n", strerror(errno));
        close(fd);
        return 1;
    }
    printf("[+] CTX id=%u\n", ctx.drawctxt_id);

    /* 3. Try map user memory to GPU */
    void *buf = mmap(NULL, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if (buf == MAP_FAILED) {
        printf("[-] mmap: %s\n", strerror(errno));
        goto cleanup;
    }
    memset(buf, 0x41, 4096);

    struct kgsl_map_user_mem map = {
        .len = 4096,
        .offset = 0,
        .hostptr = (unsigned long)buf,
        .memtype = KGSL_USER_MEM_TYPE_ADDR,
        .flags = 0,
    };
    ret = ioctl(fd, IOCTL_KGSL_MAP_USER_MEM, &map);
    if (ret < 0) {
        printf("[-] MAP USER MEM: %s\n", strerror(errno));
    } else {
        printf("[+] MAPPED: hostptr=%p gpuaddr=0x%lx len=%zu\n",
               buf, (unsigned long)map.gpuaddr, (size_t)map.len);
    }

    /* 4. Try GPU command */
    struct kgsl_command_object cmd = {
        .gpuaddr = map.gpuaddr ? map.gpuaddr : 0x1000,
        .size = 4096,
        .flags = KGSL_CMDLIST_IB,
        .id = 0,
    };
    struct kgsl_gpu_command gpu_cmd = {
        .flags = 0,
        .cmdlist = (uint64_t)(unsigned long)&cmd,
        .cmdsize = sizeof(cmd),
        .numcmds = 1,
        .objlist = 0,
        .objsize = 0,
        .numobjs = 0,
        .synclist = 0,
        .syncsize = 0,
        .numsyncs = 0,
        .context_id = ctx.drawctxt_id,
        .timestamp = 1,
    };
    ret = ioctl(fd, IOCTL_KGSL_GPU_COMMAND, &gpu_cmd);
    if (ret < 0) {
        printf("[-] GPU_COMMAND: %s\n", strerror(errno));
    } else {
        printf("[+] GPU_COMMAND OK timestamp=%u\n", gpu_cmd.timestamp);
    }

    /* 5. Check GPU timestamps */
    unsigned int ts = 0;
    ret = ioctl(fd, _IOWR(KGSL_IOC_TYPE, 0x18, unsigned int), &ts);
    if (ret < 0) {
        printf("[-] READ_TIMESTAMP: %s\n", strerror(errno));
    } else {
        printf("[+] GPU timestamp=%u\n", ts);
    }

cleanup:
    {
        struct kgsl_drawctxt_destroy d = { .drawctxt_id = ctx.drawctxt_id };
        ioctl(fd, IOCTL_KGSL_DRAWCTXT_DESTROY, &d);
    }
    close(fd);
    printf("[+] DONE\n");
    return 0;
}