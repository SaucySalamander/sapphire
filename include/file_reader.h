/**
 * @file file_reader.h
 * @brief Core file I/O utilities for reading text and binary files.
 * 
 * This module centralizes all file reading operations to ensure:
 * - Consistent error handling with LOG_ERROR()
 * - Path validation (no `.., embedded nulls, reasonable length)
 * - Proper memory management (caller owns returned buffers)
 * - Safe file size handling (size_t, not long)
 * 
 * All file I/O in the project MUST use these utilities.
 * Do NOT implement custom fopen/fread/fseek in domain modules.
 */

#ifndef FILE_READER_H
#define FILE_READER_H

#include <stddef.h>  /* size_t */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Read entire binary file into allocated buffer.
 * 
 * Safely reads a complete file with:
 * - Path validation (blocks `..`, null bytes, length checks)
 * - Secure allocation with size tracking
 * - Comprehensive error logging
 * 
 * @param filepath Path to file (must be const to prevent accidental modification)
 * @param out_buffer Pointer to receive allocated buffer pointer
 * @param out_size Pointer to receive file size in bytes
 * 
 * @return 0 on success, -1 on error (error already logged)
 * 
 * @note Caller is responsible for freeing the returned buffer with free()
 * @note For text files, buffer is NOT null-terminated (use file_read_json instead)
 */
int file_read_to_buffer(const char* filepath, char** out_buffer, size_t* out_size);

/**
 * @brief Read JSON file into allocated, null-terminated buffer.
 * 
 * Specialized for JSON files:
 * - Validates path
 * - Reads entire file
 * - Null-terminates the buffer for safe string operations
 * - Logs all errors
 * 
 * @param filepath Path to JSON file (must be const)
 * @param out_json Pointer to receive allocated JSON string pointer
 * @param out_length Pointer to receive JSON length (excluding null terminator)
 * 
 * @return 0 on success, -1 on error (error already logged)
 * 
 * @note Caller is responsible for freeing the returned buffer with free()
 * @note Buffer is null-terminated for safe string operations
 */
int file_read_json(const char* filepath, char** out_json, size_t* out_length);

/**
 * @brief Validate and construct a path from directory and component parts.
 * 
 * Validates each path component separately and safely concatenates them.
 * Blocks path traversal (..), embedded nulls, newlines, and validates total length.
 * Allocates and returns the complete path string - caller owns the returned buffer.
 * 
 * @param dir Directory base path to validate (e.g., "./models")
 * @param component1 First path component (e.g., model_name, "tokenizer", "config")
 * @param component2 Optional second component, NULL to skip (e.g., "tokenizer.json")
 * 
 * @return Newly allocated path string on success, NULL on error (error already logged)
 * 
 * @note Caller is responsible for freeing the returned pointer with free()
 * @note All error messages are logged at point of failure
 * @example char *path = construct_safe_path("./models", "gemma-3-270m-it", NULL);
 *          if (!path) return error;
 *          // use path... "./models/gemma-3-270m-it"
 *          free(path);
 * @example char *path = construct_safe_path("./models", "gemma-3-270m-it", "tokenizer.json");
 *          produces "./models/gemma-3-270m-it/tokenizer.json"
 */
char* construct_safe_path(const char* dir, const char* component1, const char* component2);

#ifdef __cplusplus
}
#endif

#endif  // FILE_READER_H
