# Bill extraction

This tool reads KE, LESCO, and IESCO electricity-bill images and writes one JSON object per bill into `level1.csv`. It accepts `.jpg`, `.jpeg`, and `.png` files only. It does not read PDFs.

The model name comes from the `.env` file. The model reads the printed page. Python then normalizes dates and amounts and applies the field rules below. Python does not recalculate units, taxes, totals, or payables.

The images in `D:\utility\train` are a check set. The code does not branch on those file names.

## How the code is put together

The work has two steps, and `level1.csv` is the link between them. The first step reads a bill photo and writes one JSON object per bill. The second step reads questions, finds that bill by `bill_id`, and writes the answer. Questions use the saved JSON. They do not open the photo again, so every answer stays on the figures that were extracted.

```text
bill image
    -> vision model reads the full page (this is the main read)
    -> Python cleans dates, amounts, and labels
    -> if a field is still empty:
           zoom a copy, and the vision model reads that copy
           if that field is still empty, Tesseract OCR reads the zoomed copy
    -> level1.csv  (bill_id, json)
            -> questions CSV, joined on bill_id
            -> scope check, then an answer from that JSON
            -> level2.csv  (bill_id, question_id, question, answer)
```

### The model reads. Python decides the final values.

The vision model reads the photo. The page is sent to it as an image, and it returns the printed fields as JSON. Tesseract does not do this first read. Python then turns that text into the final bill object.

The vision model is used for the first read because the bills are photographs, and the layout differs across KE, LESCO, and IESCO. OCR is a later backup for a field the model left empty. Python is used for the final values because a date, a credit, and a rupee amount must follow the same rules every time. The same printed text becomes the same JSON on a later run. Units, taxes, totals, and payables stay as printed. When a subtotal does not match the sum of its lines, the printed subtotal is kept and a warning is recorded.

Amounts stay exact. The model returns them as text, including commas and a CR suffix. Python stores them as decimals and writes the shortest plain number. Ordinary floating-point numbers can change a printed amount while it is saved, so those values are refused.

The model is asked for one fixed set of fields, with temperature set to 0. The fixed fields keep every bill on the same shape. Temperature 0 asks the model to stay consistent, so a second read of the same page stays close to the first. If the model rejects temperature or that fixed shape, the same request is sent once more without the rejected setting. A timeout, a rate limit, or a server error is tried up to three times, with a short wait that doubles between tries. Invalid JSON gets one repair request that must keep the printed amounts and dates. The API key is never written into a log, an error, or a session file.

### A second look fills only what the first read missed.

The full page is read first. When a field is blank or marked unreadable, the tool zooms a temporary copy and asks the model again. If that field is still missing, Tesseract reads the zoomed copy. A value the first read already returned is left as it is.

This order exists because a small figure is easier to miss on a full page than in a larger view. OCR is last because it is weaker than the model on these bills, and it is used only for a field that is still empty. The original image is never changed, and the temporary copy is deleted even when the later reads fail.

### One bad file does not stop the batch.

Each image is handled on its own. A file that cannot be opened, or whose model call fails, is left out of `level1.csv`. The reason is printed and, when there is something to report, written beside the CSV as `level1.errors.json`. A bad field on an otherwise readable bill is reported, and the rest of that bill is kept. A value that is not printed becomes null. A correct null is a valid result.

### The CSV is the join between a bill and its questions.

`level1.csv` has two columns, `bill_id` and `json`. The bill id is the image name without its extension. The whole bill stays in one cell so it can be stored, checked, and reused without opening the image. A question file uses the same `bill_id`. That is how `level2.csv` is built: each question row is matched to the JSON for that bill, and the answer is written on the same row.

Names, addresses, account numbers, and meter numbers are left out of the JSON. Answers are written from that JSON, so those personal details are not available to the answer step.

### Questions are checked, then answered from the saved bill.

Phidata runs the question step. Bill extraction does not need it, so it is installed on its own. Extraction keeps working when Phidata is absent.

A question is checked before it is answered. The check is a short model call that must say whether the question is about this bill. A question about the bill, a calculation from its figures, or an estimate based on those figures is answered. Any other topic is declined, and no answer is written. The answer then uses only that bill's JSON.

The answer model also uses temperature 0, so the same question about the same bill stays stable across runs. A calculator does the addition, subtraction, multiplication, and division. Averages, differences, and similar figures are computed there, from the numbers on the bill. When a needed figure is missing, the answer says it is not on the bill. An estimate is labeled as an estimate and includes the assumption used.

Each conversation is saved as a JSON file under `agent_sessions`, so a later question about the same bill can continue it. In a question file, each row gets its own conversation name, built from the bill id and the question id. One question does not carry another question's reply. Secret fields and key-shaped text are removed before a session file is written.

### Why these pieces were selected

| Piece | Selection | Why |
| --- | --- | --- |
| Bill reader | OpenAI vision model named in `.env`. The default is `gpt-5-mini`. | The input is a photo. The model name stays in configuration, so it can change without a code change. |
| Extraction call | The OpenAI client, one request and one JSON reply. | Reading a page does not need a conversation or tools. |
| Question call | A Phidata agent. | A question needs a saved conversation, a calculator, and a check that keeps the topic on this bill. |
| Images | Pillow opens the file. OpenCV zooms a copy. Tesseract is the last try. | Each library does one job, and a field already read is not sent through the later steps. |
| Numbers | Exact decimals, written as plain text. | A printed rupee amount must survive the round trip into `level1.csv`. |
| Files | `level1.csv` for bills, `level2.csv` for answers, joined on `bill_id`. | Extraction and answers stay separate, and a question can be rerun from the saved bill. |

## Setup

Install the package into the Python that `py` launches. You can run this from any folder:

```powershell
py -m pip install -e D:\utility\source
```

Put the key and the model name in `D:\utility\source\.env`. The tool reads that file on its own. You do not need to set the variables in the shell. Use these names:

- `MODEL_PROVIDER` for the provider. Use `openai`. When this is unset, the tool uses OpenAI.
- `OPENAI_API_KEY` for the key. If that name is absent, `API_KEY` in the same file is accepted.
- `MODEL_NAME` for the model. When this is unset, the tool uses `gpt-5-mini`.

Questions about a saved bill use the same `.env` values. Install phidata once so those questions can run:

```powershell
py -m pip install phidata==2.7.10 --no-deps
```

Do not commit `.env`. Do not paste the key into source code or into this README.

## How to test

After the install command above, this works from `D:\utility` or from `D:\utility\source`:

```powershell
py -m bill_extract.cli --input D:\utility\train --output D:\utility\source\level1.csv
```

`--input` can be one JPG or PNG image, or a folder of those images. `--output` is the CSV file to create.

Unit tests check normalization, CSV writing, the question scope check, and level2.csv answers. They do not call the API.

```powershell
cd D:\utility\source
py -m unittest discover -s tests -v
```

## Ask about a bill

This answers a customer question from one bill in `level1.csv`. The answer can explain a figure, calculate from the figures on the bill, or give an estimate based on that bill. A question about anything else is declined.

The conversation is saved as a JSON file with phidata's JSON agent storage. Ask again with the same bill to continue the same conversation.

```powershell
py -m bill_extract ask --bill D:\utility\source\level1.csv --bill-id KESC_0008 --question "How much more is due after the due date?"
```

`--bill` can also be a JSON file that contains one bill. `--bill-id` is required only when the CSV has more than one bill. `--session` sets the conversation name. When it is omitted, the bill id is used.

The command exits with `0` when you get a reply, including a reply that declines an unrelated question. It exits with `1` when the bill file or the model request fails, and with `2` when the settings are missing.

## Answer a question file

Use this when the questions are already listed in a CSV file. Each row has the bill id and the question. The bill id is the image name used when that bill was extracted into `level1.csv`. The answer is written on the same row in `level2.csv`.

```powershell
py -m bill_extract answers --bills D:\utility\source\level1.csv --questions D:\utility\test\csv\level2.csv --output D:\utility\source\level2.csv
```

The questions file must have `bill_id` and `question` columns. `question_id` is copied into the result when it is present. When `--output` is omitted, `level2.csv` is written in the same folder as the bills file.

One question that cannot be answered does not stop the others. That row still appears in `level2.csv`, with a short explanation in the answer column. The command exits with `1` when any row could not be answered.

## Exit codes

The exit code is `0` when every file succeeds, `1` when one or more files fail, and `2` when the configuration is invalid. A missing key is configuration and exits with `2` before any image is opened.

One bad image does not stop the batch. A failed file is left out of the CSV. The reason is printed and, when there is something to report, written beside the CSV as `level1.errors.json`.

A bad field is reported on that bill, and the rest of the bill is still returned.

## Zoom fallback

The full page is read first. If a field is blank or marked unreadable, the tool makes a temporary copy of the image and zooms that copy with OpenCV. The zoomed copy is sent back to the model. If a value is still missing, Tesseract reads the same zoomed image. Only the unread fields are filled. A value the full page already returned is left as it is. The original image is not changed, and it is not cropped. The temporary copy is deleted even when the later reads fail. If the zoomed image still cannot supply a field, that field stays null and the error is recorded. The rest of the bill is kept. When nothing is missing, the image is not zoomed.

Tesseract must be installed for the OCR step. The Python package is `pytesseract`. On Windows, install the Tesseract program and make sure `tesseract` is on the PATH.

## JSON fields

Each bill object uses numbers, not quoted amounts.

- `provider` is KE, LESCO, or IESCO, taken from the bill header. A KESC file is provider KE.
- `tariff` is copied as printed.
- `sanctioned_load_kw` comes from Sanc Load, S.Load, or San Load.
- `bill_month` is `YYYY-MM`. `reading_date`, `issue_date`, and `due_date` are `YYYY-MM-DD`. The due date is the last date to pay with no late-payment surcharge.
- When several meter rows are printed and a row has its own date or time, `previous_reading` and `current_reading` come from the latest row. Rows are not added together. A row with no date is not treated as the latest.
- `charges` has one entry per printed charge line. Blank lines and zero amounts are skipped. The same type can appear twice, so two energy lines stay two entries. Types include `fixed`, `energy`, `quarterly_adjustment`, `fpa`, and `surcharge`.
- `taxes` has one entry per line in the tax section. Types include `electricity_duty`, `gst`, and `municipal_tax`. If the bill prints only one combined tax amount, `taxes` is `[]` and that amount is `total_taxes`.
- `current_bill` is this month's printed bill, not Amount Due.
- `arrears` keeps the printed sign. A credit of 0.53 is `-0.53`. A blank value is null. A printed 0 stays 0. When arrears is printed as an amount and a month count separated by `/` or `\`, only the amount is kept. `39411 / 2` is `39411`.
- `payable_after_due_date` is the highest printed late-payment payable total. It is not a surcharge component, and it is not a computed sum.

A subtotal that does not match the sum of its lines is kept as printed. The result records a warning and does not replace it.

## CSV quoting

`level1.csv` has two columns, `bill_id` and `json`. `bill_id` is the file name without its extension. The JSON object is one line inside the CSV cell. The CSV writer quotes that cell and doubles every quotation mark inside it:

```text
bill_id,json
KESC_0008,"{""provider"":""KE"",""tariff"":""A1-R"",""sanctioned_load_kw"":3, ... }"
```
