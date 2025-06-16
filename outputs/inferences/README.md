# Readme on inferences

## Decryption

Since certain outputs are deemed harmful, we guard the use of these files in the same as jailbreaking adapters.

See ../../reft-adapters/README.md on additional agreements and instructions to decrypt.

By decrypting, you agree to respect the same use restrictions listed in the agreement.

## Folder Format

For example:

```
inferences_bbh/multistep_arithmetic_two_TRAINEDON_OpenMathReasoning_Llama-3.1-8B-Instruct_TO_Llama-3.2-3B-Instruct_WITH_NodireftIntervention/inference_results_linear_1024_PinvConverter_0;2;4;6;8;10;12;14;16;18;20;22;24;26_last.csv
```

is

| Key                    | Value                                   |
|------------------------|-----------------------------------------|
| Dataset / SubTask      | inferences_bbh/multistep_arithmetic_two |
| Training Dataset       | TRAINEDON_OpenMathReasoning_            |
| Donor Model            | Llama-3.1-8B-Instruct                   |
| Recipient Model        | Llama-3.2-3B-Instruct                   |
| ReFT Intervention      | NodireftIntervention                    |
| Layer Matching         | Linear                                  |
| Max new token          | 1024                                    |
| Converter              | PinvConverter                           |
| Intervened Donor Layer | 0;2;4;6;8;10;12;14;16;18;20;22;24;26    |
| Activation token used  | last prompt token                       |
