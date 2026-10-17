# Bin-Sight 🔬

Snap a photo of lab waste, get told which bin it goes in, and log the disposal on Solana.

Built at the IKU Womxn in STEM Hackathon 2026.

## The problem
New students in university labs often don't know which waste stream an item belongs to: sharps, biohazard, halogenated vs non-halogenated solvents, contaminated solids. Getting it wrong is a safety and compliance risk, and hazardous waste needs a reliable disposal record.

## How it works
1. **Choose your bins.** Every lab is different, so you tick the bins your lab actually has (hydrocarbon, chlorinated, halogenated, aqueous, silver nitrate, heavy metal, silica, autoclave, sharps, recycling…) and can add your own specialist bins. The choice is remembered on that device.
2. **Take a photo** in the browser (works on phones) and/or **describe it** in a few words ("lead nitrate solution"). A description tells Gemini what a photo can't show, like what's inside a bottle; description-only works too.
3. **Where to throw it.** **Gemini** (`gemini-3.8-flash`) looks at the image and picks one of *your* bins, returning structured JSON: bin, confidence, hazards, PPE and reasoning. It always gives a decisive answer, so students don't have to queue up to ask the technician. If the photo is unclear it still answers, flags it, and suggests a second-choice bin to check against the label. If the waste needs a bin your lab hasn't selected, it never substitutes a "close enough" bin; it names the bin needed, and one tap adds it. When the user confirms, the backend writes a compact record (bin, category, timestamp, image hash) to **Solana devnet** with the Memo program, giving a tamper-proof disposal log.
4. **How to clean it (optional).** If the photo shows reusable equipment such as a flask or spatula, Bin-Sight gives step-by-step cleaning instructions, including which of your bins the rinse goes into.

## The waste policy decides, not the AI
Bin-Sight is grounded in a lab waste policy spreadsheet (`data/lab_waste_taxonomy.xlsx`): 85 lab items, 35 waste categories, handling steps, "Do NOT" rules and 15 chemical mixing rules.

1. Gemini identifies what's in the photo and matches it to an item in the policy (e.g. *I042 Chloroform*).
2. **The policy rules then choose the bin**: item → policy category → the bin your lab has. Gemini cannot override this.
3. The student sees the policy's own handling steps, its "Do NOT" warning, and any mixing rules that apply (e.g. *never mix azide with acids: forms toxic, explosive gas*).
4. Only when nothing in the policy matches does Gemini decide alone, and the screen says so.

Every answer is labelled "Decided by your lab's waste policy" or "Gemini's best judgement". To adapt Bin-Sight to another lab, edit the spreadsheet and run `python import_taxonomy.py`.

Radioactive, animal and high-containment waste, plus unlabelled unknowns, are never assigned to a bin: they have their own legal procedures, and Bin-Sight says so.

## Measuring accuracy
`python eval.py eval_photos` runs labelled photos (in folders named after the correct bin) through the classifier and produces `eval_report.html`: accuracy, a confusion matrix, policy-vs-AI accuracy, and every mistake split into "wrong and confident" vs "wrong but warned".

## Tech stack
- Backend: Python, FastAPI
- AI: Google Gemini API with structured output, grounded in a spreadsheet-based waste policy
- Blockchain: Solana devnet, Memo program (`solders` to sign, JSON-RPC to send)
- Frontend: plain HTML, CSS and JavaScript

## Run it
```bash
python -m pip install -r requirements.txt
cp .env.example .env          # then paste your Gemini key into .env
python import_taxonomy.py     # only needed after editing the policy spreadsheet
python solana_log.py          # creates a devnet wallet and requests test SOL
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
Open http://localhost:8000.

Waste categories live in `waste_categories.json`, so any lab can adapt them to its own rules.

## AI usage disclosure
Gemini performs the image classification at runtime. Parts of the code were drafted with help from an AI coding assistant and then tested and adapted by the team.

## Safety note
Bin-Sight never puts waste in an unsuitable bin: if a lab doesn't have the right container, it says so. It is designed to answer the routine "which bin?" questions that take up technicians' time; labs should still follow their local waste rules.
