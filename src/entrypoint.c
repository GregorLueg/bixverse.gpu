// We need to forward routine registration from C to Rust
// to avoid the linker removing the static library.

void R_init_bixverse_gpu_extendr(void *dll);
void register_extendr_panic_hook(void);

void R_init_bixverse_gpu(void *dll) {
    register_extendr_panic_hook();
    R_init_bixverse_gpu_extendr(dll);
}
