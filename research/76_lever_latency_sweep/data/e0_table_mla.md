# E0 infrastructure-parity table

| arm | attn | cudagraph | moe_backend | prepare_finalize | quant | extra | verdict | notes |
|---|---|---|---|---|---|---|---|---|
| ds_bf16 | FLASH_ATTN_MLA | FULL_AND_PIECEWISE | TRITON | MoEPrepareAndFinalizeNaiveDPEPModu | None | - | PASS |  |
| ds_fp8marlin | FLASH_ATTN_MLA | FULL_AND_PIECEWISE | MARLIN | MoEPrepareAndFinalizeNaiveDPEPModu | fp8 | - | INFO:OK |  |
| ds_fp8block | FLASH_ATTN_MLA | FULL_AND_PIECEWISE | FLASHINFER_CUTLASS | MoEPrepareAndFinalizeNaiveDPEPModu | fp8_per_block | - | INFO:OK |  |
| ds_kvq | FLASHMLA | FULL_AND_PIECEWISE | TRITON | MoEPrepareAndFinalizeNaiveDPEPModu | None | - | INFO:BROKEN-CELL | attn FLASHMLA != ref FLASH_ATTN_MLA |
| ds_win | FLASH_ATTN_MLA | FULL_AND_PIECEWISE | TRITON | MoEPrepareAndFinalizeNaiveDPEPModu | None | sliding_window=512 num_hidden_layers=2 | INFO:OK |  |
| ds_bf16dummy | FLASH_ATTN_MLA | FULL_AND_PIECEWISE | TRITON | MoEPrepareAndFinalizeNaiveDPEPModu | None | - | PASS |  |
| ds_skip50 | FLASH_ATTN_MLA | FULL_AND_PIECEWISE | TRITON | MoEPrepareAndFinalizeNaiveDPEPModu | None | sliding_window=None num_hidden_layers= | PASS |  |

GATE: OPEN (group=mla)
