/**
 * @file file_reader.c
 * @brief Core file I/O utilities implementation.
 * 
 * Centralizes file reading with:
 * - Path validation and sanitization
 * - Safe memory allocation and error handling
 * - Consistent logging with LOG_ERROR()
 * - size_t for file sizes (supports >2GB files)
 */

#include "file_reader.h"
#include "log.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * @brief Validate file path for safety.
 * 
 * Checks for:
 * - NULL pointer
 * - Path traversal attempts (..)
 * - Embedded null bytes
 * - Newline characters
 * - Reasonable length (max 4096)
 * 
 * @param filepath Path to validate
 * @return 0 if valid, -1 if invalid (error already logged)
 */
static int validate_filepath(const char* filepath) {
    if (!filepath) {
        LOG_ERROR("filepath is NULL");
        return -1;
    }

    // Check for path traversal
    if (strstr(filepath, "..") != NULL) {
        LOG_ERROR("filepath contains '..' (path traversal attempt): %s", filepath);
        return -1;
    }

    // Check for embedded null bytes (beyond the string's own terminator)
    int null_count = 0;
    for (size_t i = 0; i <= strlen(filepath); i++) {
        if (filepath[i] == '\0') null_count++;
    }
    if (null_count > 1) {
        LOG_ERROR("filepath contains embedded null bytes");
        return -1;
    }

    // Check for newline characters
    if (strchr(filepath, '\n') != NULL || strchr(filepath, '\r') != NULL) {
        LOG_ERROR("filepath contains newline characters");
        return -1;
    }

    // Check reasonable length (typically PATH_MAX is 4096)
    size_t filepath_len = strlen(filepath);
    const size_t MAX_PATH_LEN = 4096;
    if (filepath_len > MAX_PATH_LEN) {
        LOG_ERROR("filepath too long (%zu bytes, max %zu)", filepath_len, MAX_PATH_LEN);
        return -1;
    }

    return 0;
}

/**
 * @brief Validate and construct a path from directory and component parts.
 * 
 * Validates each path component separately, calculates exact size needed,
 * allocates a buffer, and safely concatenates them with forward slashes.
 * 
 * @param dir Directory base path to validate
 * @param component1 First path component (required)
 * @param component2 Optional second component, NULL to skip
 * 
 * @return Newly allocated path string on success, NULL on error (all errors pre-logged)
 * 
 * @note Caller is responsible for freeing the returned pointer with free()
 */
char* construct_safe_path(const char* dir, const char* component1, const char* component2) {
    if (!dir || !component1) {
        LOG_ERROR("Invalid arguments to construct_safe_path");
        return NULL;
    }
    
    // Validate each component separately
    if (validate_filepath(dir) != 0) return NULL;
    if (validate_filepath(component1) != 0) return NULL;
    if (component2 && validate_filepath(component2) != 0) return NULL;
    
    // Calculate exact size needed: dir + '/' + comp1 + ['/'] + [comp2] + null
    size_t total_size = strlen(dir) + 1 + strlen(component1) + 1;
    if (component2) {
        total_size += strlen(component2);
    }
    
    // Allocate exact size needed
    char* path = (char*)malloc(total_size);
    if (!path) {
        LOG_ERROR("Cannot allocate memory for path (%zu bytes)", total_size);
        return NULL;
    }
    
    // Construct path with components
    int rc;
    if (component2) {
        rc = snprintf(path, total_size, "%s/%s/%s", dir, component1, component2);
    } else {
        rc = snprintf(path, total_size, "%s/%s", dir, component1);
    }
    
    if (rc < 0 || (size_t)rc >= total_size) {
        LOG_ERROR("Path construction failed");
        free(path);
        return NULL;
    }

    LOG_DEBUG("Constructed path: %s", path);
    return path;
}

int file_read_to_buffer(const char* filepath, char** out_buffer, size_t* out_size) {
    if (!filepath || !out_buffer || !out_size) {
        LOG_ERROR("Invalid arguments: filepath=%p out_buffer=%p out_size=%p", filepath, out_buffer, out_size);
        return -1;
    }

    // Initialize output pointers
    *out_buffer = NULL;
    *out_size = 0;

    // Validate path
    if (validate_filepath(filepath) != 0) {
        return -1;  // Error already logged
    }

    // Open file
    FILE* f = fopen(filepath, "rb");
    if (!f) {
        LOG_ERROR("Cannot open file: %s", filepath);
        return -1;
    }

    // Get file size
    if (fseek(f, 0, SEEK_END) != 0) {
        LOG_ERROR("Cannot seek to end of file: %s", filepath);
        fclose(f);
        return -1;
    }

    long file_size_long = ftell(f);
    if (file_size_long < 0) {
        LOG_ERROR("Cannot determine file size: %s", filepath);
        fclose(f);
        return -1;
    }

    size_t file_size = (size_t)file_size_long;

    // Sanity check: prevent allocation of excessively large files
    // (adjust as needed for your use case; 1GB is reasonable)
    const size_t MAX_FILE_SIZE = 1024 * 1024 * 1024;  // 1 GB
    if (file_size > MAX_FILE_SIZE) {
        LOG_ERROR("File too large (%zu bytes, max %zu): %s", file_size, MAX_FILE_SIZE, filepath);
        fclose(f);
        return -1;
    }

    if (fseek(f, 0, SEEK_SET) != 0) {
        LOG_ERROR("Cannot seek to start of file: %s", filepath);
        fclose(f);
        return -1;
    }

    // Allocate buffer (for binary, no extra space for null terminator)
    char* buffer = (char*)malloc(file_size);
    if (!buffer) {
        LOG_ERROR("Cannot allocate memory for file (%zu bytes): %s", file_size, filepath);
        fclose(f);
        return -1;
    }

    // Read file
    size_t bytes_read = fread(buffer, 1, file_size, f);
    if (bytes_read != file_size) {
        LOG_ERROR("Failed to read file (read %zu of %zu bytes): %s", bytes_read, file_size, filepath);
        free(buffer);
        fclose(f);
        return -1;
    }

    fclose(f);

    *out_buffer = buffer;
    *out_size = file_size;

    LOG_DEBUG("Successfully read file: %s (%zu bytes)", filepath, file_size);
    return 0;
}

int file_read_json(const char* filepath, char** out_json, size_t* out_length) {
    if (!filepath || !out_json || !out_length) {
        LOG_ERROR("Invalid arguments: filepath=%p out_json=%p out_length=%p", filepath, out_json, out_length);
        return -1;
    }

    // Initialize output pointers
    *out_json = NULL;
    *out_length = 0;

    // Validate path
    if (validate_filepath(filepath) != 0) {
        return -1;  // Error already logged
    }

    // Open file
    FILE* f = fopen(filepath, "rb");
    if (!f) {
        LOG_ERROR("Cannot open JSON file: %s", filepath);
        return -1;
    }

    // Get file size
    if (fseek(f, 0, SEEK_END) != 0) {
        LOG_ERROR("Cannot seek to end of JSON file: %s", filepath);
        fclose(f);
        return -1;
    }

    long file_size_long = ftell(f);
    if (file_size_long < 0) {
        LOG_ERROR("Cannot determine JSON file size: %s", filepath);
        fclose(f);
        return -1;
    }

    size_t file_size = (size_t)file_size_long;

    // Sanity check for JSON files (adjust if needed; 100MB is reasonable)
    const size_t MAX_JSON_SIZE = 100 * 1024 * 1024;  // 100 MB
    if (file_size > MAX_JSON_SIZE) {
        LOG_ERROR("JSON file too large (%zu bytes, max %zu): %s", file_size, MAX_JSON_SIZE, filepath);
        fclose(f);
        return -1;
    }

    if (fseek(f, 0, SEEK_SET) != 0) {
        LOG_ERROR("Cannot seek to start of JSON file: %s", filepath);
        fclose(f);
        return -1;
    }

    // Allocate buffer with extra space for null terminator
    char* buffer = (char*)malloc(file_size + 1);
    if (!buffer) {
        LOG_ERROR("Cannot allocate memory for JSON file (%zu bytes): %s", file_size, filepath);
        fclose(f);
        return -1;
    }

    // Read file
    size_t bytes_read = fread(buffer, 1, file_size, f);
    if (bytes_read != file_size) {
        LOG_ERROR("Failed to read JSON file (read %zu of %zu bytes): %s", bytes_read, file_size, filepath);
        free(buffer);
        fclose(f);
        return -1;
    }

    // Null-terminate for safe string operations
    buffer[file_size] = '\0';

    fclose(f);

    *out_json = buffer;
    *out_length = file_size;

    LOG_DEBUG("Successfully read JSON file: %s (%zu bytes)", filepath, file_size);
    return 0;
}
