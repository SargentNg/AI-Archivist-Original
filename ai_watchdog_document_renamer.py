import os
import re
import shutil
import time
import base64

import fitz  # PyMuPDF
import requests
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

# --- CONFIGURATION ---
WATCH_FOLDER = r"C:\\Users\\sarge\\OneDrive\\Documents\\Unsorted Docs"
OUTPUT_FOLDER = r"C:\\Users\\sarge\\OneDrive\\Family Vault\\Scans"
API_URL = "http://192.168.50.20:11434/api/generate"
MODEL = "gemma3:12b"
MODEL2 = "gemma3:12b"

SYSTEM_PROMPT = """
You are an expert Data Archivist for a high-net-worth individual.
Your goal is to look at the provided document image and output a precise filename for sorting.

NAMING CONVENTION:
Entity_Detail_DocType_YYYY-MM-DD[optional: _to_YYYY-MM-DD].pdf

EXTRACTION RULES:
1. ENTITY: The company, bank, or person issuing the doc (e.g., Schwab, IRS, HomeDepot, tenants). 
   - Prefer "Etrade" if Etrade (Do NOT use Morgan Stanley if its an Etrade account). 
   - Prefer "Schwab" instead of CharlesSchwab.
   - Prefer "Foxbury Park HOA" instead of "Foxbury Park"
   - Prefer "Freedom Mortgage" instead of "FreedomMortgage"
   - Prefer "Hawaii Gas" for Hawaii Gas. 
   - Prefer "YYYY 1099" for 1099 tax forms
   - Prefer "YYYY 1098" for 1099 tax forms
2. DETAIL: Account Type [optional: and Owner if not Jonathan] and last 4 digits of Account number, Invoice number, or Property Name. Keep it short.
   - For Etrade: If you see "Traditional IRA", "Roth IRA", or "Rollover IRA", use that.
   - Account Numbers: ALWAYS extract only the final 4 digits.
   - Account Type: For credit card accounts, output the type of card program and card network. Example: BestBuy Mastercard
   - For Trade Confirmations, after the Trade_Confirmation text, include the following: Buy(s) or Sell(s) and #(s) of shares and ticker(s)
   - Owner: Include Owner if it's Peter, Angela, Anna, Wyatt, Warren but not Jonathan
   - For Hawaiian Electric Invoices, output the account number and service address
   - For Hawaii Gas Invoices, output the service address and account number
3. DOCTYPE: Invoice, Statement, Mortgage_Statement, Receipt, Tax_Form, Contract, Report, Trade_Confirmation, Escrow_Account_Review, 1099-MISC, 1098, 1099-INT.
4. DATE: Prefer "Letter Date" or "Statement Date" or "Invoice Date". Prefer Service Period dates for Utility bills or beginning and end dates for policy periods. If not found, use "Due Date". 
   - OPTIONAL END DATE: If explicitly covering a date range, append the end date using "_to_YYYY-MM-DD".
   - FORMAT: You MUST output dates in strict "YYYY-MM-DD" or "YYYY-MM-DD_to_YYYY-MM-DD".
5. SECURITY RULE: Never output full account numbers. Digits, only last 4 digits.

OUTPUT INSTRUCTIONS:
- Output ONLY the filename.
- Do not use spaces (use underscores).
- Do not include explanations, markdown formatting, or introductory text.
- Do not include any characters not allowed in filenames
- Use sentence case except for names that are not sentence case
"""

class AI_Renamer_Handler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory or not event.src_path.lower().endswith(".pdf"):
            return
        print(f"\\n[+] Detected new file: {event.src_path}")
        if self.wait_for_file_ready(event.src_path):
            if not os.path.exists(event.src_path):
                return
            try:
                self.process_pdf(event.src_path)
            except Exception as e:
                print(f"[!] Critical Error: {e}")

    def wait_for_file_ready(self, file_path, timeout=600):
        print("    Waiting for file write...", end="", flush=True)
        historical_size = -1
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                with open(file_path, "rb"):
                    pass
                current_size = os.path.getsize(file_path)
                if current_size == historical_size and current_size > 0:
                    print(" Done.")
                    return True
                historical_size = current_size
                time.sleep(1)
            except OSError:
                time.sleep(1)
        print(" Timeout.")
        return False

    def compress_date_range(self, filename_no_ext: str) -> str:
        def replace_same_year(match):
            y1, m1, d1 = match.group(1), match.group(2), match.group(3)
            y2, m2, d2 = match.group(4), match.group(5), match.group(6)
            if y1 == y2:
                return f"{y1}-{m1}-{d1}_to_{m2}-{d2}"
            return match.group(0)
        return re.sub(
            r"(\\d{4})-(\\d{2})-(\\d{2})_to_(\\d{4})-(\\d{2})-(\\d{2})",
            replace_same_year,
            filename_no_ext
        )

    def clean_filename(self, filename):
        filename = filename.replace("```", "").strip()
        if filename.lower().endswith(".pdf"):
            filename = filename[:-4]
        filename = filename.replace(" ", "_")
        filename = re.sub(r"[^a-zA-Z0-9_\\-]", "", filename)
        filename = self.compress_date_range(filename)
        return filename[:100] + ".pdf"

    def is_garbage_text(self, raw_text: str) -> bool:
        """
        Analyzes the statistical distribution of the text layer to determine
        if it is a pristine digital document or a bad scanner OCR layer.
        """
        if len(raw_text) < 150:
            return True  # Too short to be a reliable digital page

        # 1. Alphanumeric Ratio Test
        # Digital statements are usually 75%+ alphanumeric.
        alnum_count = sum(c.isalnum() for c in raw_text)
        alnum_ratio = alnum_count / len(raw_text)

        # 2. Word Fragmentation Test
        # Digital English/Financial text averages 4.5 to 5.5 chars per word.
        # Bad OCR often drops below 3.0 due to excessive injected spaces.
        words = raw_text.split()
        avg_word_length = sum(len(w) for w in words) / len(words) if words else 0

        # Debug print so you can monitor the math in the console
        print(f"    [🔍 Text Stats] Alphanumeric Ratio: {alnum_ratio:.2f} | Avg Word Length: {avg_word_length:.2f}")

        # If it fails the baseline thresholds of a normal digital document, it's garbage.
        if alnum_ratio < 0.65 or avg_word_length < 3.0:
            return True
            
        return False


    def process_pdf(self, file_path):
        print("    Analyzing PDF type...")
        try:
            doc = fitz.open(file_path)
            page = doc.load_page(0) 
            raw_text = page.get_text("text").strip()
            
            # --- USE THE NEW HEURISTIC HERE ---
            if not self.is_garbage_text(raw_text):
                print("    [!] Clean digital text detected. Bypassing Vision model.")
                text_preview = raw_text[:3000] 
                full_prompt = f"{SYSTEM_PROMPT}\\n\\nDOCUMENT CONTENT:\\n{text_preview}\\n\\nFILENAME:"
                payload = {
                    "model": MODEL,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                    "temperature": 0.0,
                    "top_p": 0.8,
                    "top_k": 20,
                    "min_p": 0.0,
                    "presence_penalty": 1.5,
                    "repetition_penalty": 1.0,
                    "num_ctx": 4096,
                    # "num_gpu": 35,
                    # "num_thread": 16,
                    }
                }
            else:
                print("    [!] Scanned/Bad OCR detected. Engaging Llama 4 Vision model.")
                pix = page.get_pixmap(dpi=200)
                img_data = pix.tobytes("jpeg")
                img_base64 = base64.b64encode(img_data).decode('utf-8')
                payload = {
                    "model": MODEL2,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                    "temperature": 0.0,
                    "top_p": 0.8,
                    "top_k": 20,
                    "min_p": 0.0,
                    "presence_penalty": 1.5,
                    "repetition_penalty": 1.0,
                    "num_ctx": 4096,
                    }
                }
                
            doc.close()
            
        except Exception as e:
            print(f"    [!] PDF Parsing Failed: {e}")
            return

        print("    Querying AI Agent...")
        try:
            # The API call remains exactly the same, we just pass the dynamically chosen payload
            response = requests.post(
                API_URL,
                json=payload,
                timeout=300, 
            )
            
            if response.status_code != 200:
                print(f"    [!] API Error: {response.status_code} - {response.text}")
                return

            raw_name = response.json().get("response", "").strip()
            print(f"    [Raw LLM]: {raw_name}")


            final_name = self.clean_filename(raw_name)
            os.makedirs(OUTPUT_FOLDER, exist_ok=True)
            new_path = os.path.join(OUTPUT_FOLDER, final_name)

            counter = 1
            base_name = final_name[:-4]
            while os.path.exists(new_path):
                new_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_{counter}.pdf")
                counter += 1

            shutil.move(file_path, new_path)
            print(f"    [SUCCESS] Moved to: {new_path}")

        except Exception as e:
            print(f"    [!] Logic Error: {e}")


if __name__ == "__main__":
    try: 
        if not os.path.exists(WATCH_FOLDER):
            print(f"[!] Watch folder missing: {WATCH_FOLDER}")
            raise SystemExit(1)

        observer = Observer()
        event_handler = AI_Renamer_Handler()
        observer.schedule(AI_Renamer_Handler(), WATCH_FOLDER, recursive=False)

        print("--- AI ARCHIVIST ACTIVE (VISION MODE) ---")
        print(f"Model: {MODEL}")
        print(f"Vision Model: {MODEL2}")
        print(f"Watching: {WATCH_FOLDER}")
        print(f"Output: {OUTPUT_FOLDER}")
        print("Ctrl+C to stop.")

        observer.start()
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        observer.stop()
        observer.join()

    except Exception as e:
        import traceback
        traceback.print_exc()
        input("\\n[FATAL ERROR] Press Enter to close this window...")