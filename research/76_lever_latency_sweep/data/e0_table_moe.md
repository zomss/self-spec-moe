# E0 infrastructure-parity table

| arm | attn | cudagraph | moe_backend | prepare_finalize | quant | extra | verdict | notes |
|---|---|---|---|---|---|---|---|---|
| m_bf16 | FLASH_ATTN | FULL_AND_PIECEWISE | TRITON | MoEPrepareAndFinalizeNaiveDPEPModu | None | - | PASS |  |
| m_fp8marlin | FLASH_ATTN | FULL_AND_PIECEWISE | MARLIN | MoEPrepareAndFinalizeNaiveDPEPModu | fp8 | - | PASS |  |
| m_fp8block | FLASH_ATTN | FULL_AND_PIECEWISE | FLASHINFER_CUTLASS | MoEPrepareAndFinalizeNaiveDPEPModu | fp8_per_block | - | PASS |  |
| m_kvq | FLASH_ATTN | FULL_AND_PIECEWISE | TRITON | MoEPrepareAndFinalizeNaiveDPEPModu | None | - | PASS |  |
| m_win | FLASH_ATTN | FULL_AND_PIECEWISE | TRITON | MoEPrepareAndFinalizeNaiveDPEPModu | None | sliding_window=512 num_hidden_layers=4 | PASS |  |
| m_bf16dummy | FLASH_ATTN | FULL_AND_PIECEWISE | TRITON | MoEPrepareAndFinalizeNaiveDPEPModu | None | - | PASS |  |
| m_skip50 | FLASH_ATTN | FULL_AND_PIECEWISE | TRITON | MoEPrepareAndFinalizeNaiveDPEPModu | None | sliding_window=None num_hidden_layers= | PASS |  |
| m_localroute | FLASH_ATTN | FULL_AND_PIECEWISE | TRITON | MoEPrepareAndFinalizeNaiveDPEPModu | None | LOCAL_ROUTE engaged | PASS |  |
| m_skipa2a | FLASH_ATTN | FULL_AND_PIECEWISE | TRITON | MoEPrepareAndFinalizeNaiveDPEPModu | None | SKIP_A2A engaged | PASS |  |

GATE: OPEN (group=moe)
