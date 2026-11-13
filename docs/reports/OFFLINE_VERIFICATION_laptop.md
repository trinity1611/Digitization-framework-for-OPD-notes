# Offline verification report

Checked at 2026-11-13T11:15:25+00:00.

- Isolation: process-level guard only (the machine itself still has network access).
- Network probe: {'reachable': True, 'target': 'huggingface.co:443'}.
- Network use attempted by the application: none.
- **Offline inference verified for: medgemma_1_5_4b**

| step | result | detail |
|---|---|---|
| load_master_prompt | ok | sha256 d85b9b4ad985 |
| printed_text_exclusion_rule_present | ok |  |
| store_document_locally | ok | EXP_0002 |

| model | mock | local files | processor offline | inference | offline inference verified |
|---|---|---|---|---|---|
| medgemma_1_5_4b | no | complete | ok: 256 image tokens, 4729 input tokens | complete | yes |
