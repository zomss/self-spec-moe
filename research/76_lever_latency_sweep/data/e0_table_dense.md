# E0 infrastructure-parity table

| arm | attn | cudagraph | moe_backend | prepare_finalize | quant | extra | verdict | notes |
|---|---|---|---|---|---|---|---|---|
| d_bf16 | FLASH_ATTN | FULL_AND_PIECEWISE | - | - | None | - | PASS |  |
| d_w4auto | FLASH_ATTN | FULL_AND_PIECEWISE | - | - | compressed-tensors | MacheteLinearKernel | PASS |  |
| d_w4marlin | FLASH_ATTN | FULL_AND_PIECEWISE | - | - | compressed-tensors | MarlinLinearKernel | PASS |  |
| d_fp8w8a8 | FLASH_ATTN | FULL_AND_PIECEWISE | - | - | fp8 | - | PASS |  |
| d_kvq | FLASH_ATTN | FULL_AND_PIECEWISE | - | - | None | - | PASS |  |
| d_kvq_e5m2 | FLASHINFER | FULL_AND_PIECEWISE | - | - | None | - | PASS |  |
| d_win | FLASH_ATTN | FULL_AND_PIECEWISE | - | - | None | sliding_window=512 num_hidden_layers=2 | PASS |  |
| d_bf16dummy | FLASH_ATTN | FULL_AND_PIECEWISE | - | - | None | - | PASS |  |
| d_skip50 | FLASH_ATTN | FULL_AND_PIECEWISE | - | - | None | sliding_window=None num_hidden_layers= | PASS |  |

GATE: OPEN (group=dense)
