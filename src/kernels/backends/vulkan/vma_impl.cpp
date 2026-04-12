#include <stdlib.h>

static void *sapphire_vma_aligned_malloc(size_t size, size_t alignment)
{
	void *ptr = NULL;

	if (alignment < sizeof(void *)) {
		alignment = sizeof(void *);
	}

	if (posix_memalign(&ptr, alignment, size) != 0) {
		return NULL;
	}

	return ptr;
}

static void sapphire_vma_aligned_free(void *ptr)
{
	free(ptr);
}

#define VMA_SYSTEM_ALIGNED_MALLOC(size, alignment) sapphire_vma_aligned_malloc((size), (alignment))
#define VMA_SYSTEM_ALIGNED_FREE(ptr) sapphire_vma_aligned_free((ptr))
#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"
