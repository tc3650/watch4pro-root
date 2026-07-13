#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <errno.h>

/* ION */
#define ION_IOC_MAGIC 'I'
struct ion_heap_data {
    char name[32];
    uint32_t type;
    uint32_t heap_id;
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t reserved2;
};
struct ion_heap_query {
    uint32_t cnt;
    uint32_t reserved0;
    uint64_t heaps;
    uint32_t reserved1;
    uint32_t reserved2;
};
#define ION_IOC_HEAP_QUERY _IOWR(ION_IOC_MAGIC, 8, struct ion_heap_query)

struct ion_allocation_data {
    uint64_t len;
    uint32_t heap_id_mask;
    uint32_t flags;
    uint32_t fd;
    uint32_t unused;
};
#define ION_IOC_ALLOC _IOWR(ION_IOC_MAGIC, 0, struct ion_allocation_data)

/* DRI */
struct drm_version {
    int version_major;
    int version_minor;
    int version_patchlevel;
    size_t name_len;
    char *name;
    size_t date_len;
    char *date;
    size_t desc_len;
    char *desc;
};
#define DRM_IOCTL_BASE 'd'
#define DRM_IOWR(n,t) _IOWR(DRM_IOCTL_BASE, n, t)
#define DRM_IOCTL_VERSION DRM_IOWR(0x00, struct drm_version)

int main() {
    /* == ION: Query heaps == */
    printf("== ION Heap Query ==\n");
    int ion_fd = open("/dev/ion", O_RDWR);
    if (ion_fd >= 0) {
        struct ion_heap_data heaps[8];
        struct ion_heap_query q = { .cnt = 8, .heaps = (uint64_t)(unsigned long)heaps };
        int ret = ioctl(ion_fd, ION_IOC_HEAP_QUERY, &q);
        if (ret) {
            printf("HEAP_QUERY: %s\n", strerror(errno));
        } else {
            printf("Found %u heaps:\n", q.cnt);
            for (uint32_t i = 0; i < q.cnt; i++) {
                printf("  [%u] name='%s' type=%u id=%u\n", i, heaps[i].name,
                       heaps[i].type, heaps[i].heap_id);
            }
        }
        
        /* Try allocation with different heaps */
        uint32_t test_masks[] = { 0x0, 0x1, 0xffffffff, 0x2, 0x4, 0x8 };
        for (int i = 0; i < 6; i++) {
            struct ion_allocation_data d = { .len = 4096, .heap_id_mask = test_masks[i], .flags = 0 };
            int r = ioctl(ion_fd, ION_IOC_ALLOC, &d);
            printf("  alloc(mask=0x%x): %s fd=%d\n", test_masks[i], r ? strerror(errno) : "OK", d.fd);
            if (!r) { close(d.fd); break; }
        }
        close(ion_fd);
    } else {
        printf("ION open: %s\n", strerror(errno));
    }

    /* == DRI: Check render node == */
    printf("\n== DRI Render Node ==\n");
    int dri_fd = open("/dev/dri/renderD128", O_RDWR);
    if (dri_fd >= 0) {
        printf("renderD128 opened OK\n");
        char buf[256] = {};
        struct drm_version ver = { .name = buf, .name_len = sizeof(buf) };
        int ret = ioctl(dri_fd, DRM_IOCTL_VERSION, &ver);
        if (ret) printf("VERSION: %s\n", strerror(errno));
        else printf("DRM: %d.%d.%d name='%s'\n", ver.version_major, ver.version_minor,
                     ver.version_patchlevel, buf);
        close(dri_fd);
    } else {
        printf("renderD128: %s\n", strerror(errno));
    }
    
    dri_fd = open("/dev/dri/card0", O_RDWR);
    if (dri_fd >= 0) {
        printf("card0 opened OK\n");
        char buf[256] = {};
        struct drm_version ver = { .name = buf, .name_len = sizeof(buf) };
        int ret = ioctl(dri_fd, DRM_IOCTL_VERSION, &ver);
        if (ret) printf("VERSION: %s\n", strerror(errno));
        else printf("DRM: %d.%d.%d name='%s'\n", ver.version_major, ver.version_minor,
                     ver.version_patchlevel, buf);
        close(dri_fd);
    } else {
        printf("card0: %s\n", strerror(errno));
    }

    return 0;
}