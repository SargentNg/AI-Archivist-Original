import time
import os
import re
import requests
import fitz  # PyMuPDF
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# --- CONFIGURATION ---
# WATCH_FOLDER = r"C:\Users\sarge\Documents\Unsorted"
WATCH_FOLDER = r"\\RT-BE96U-EC70\HPScan"
OUTPUT_FOLDER = r"C:\Users\sarge\OneDrive\Documents\Unsorted Docs"
API_URL = "http://192.168.50.20:11434/api/generate"
# Use a smaller/faster model for this decision if possible, but Scout is fine
MODEL = "gemma3:12b" 

class PdfSplitHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory or not event.src_path.lower().endswith('.pdf'):
            return

        print(f"\n[+] Detected new scan: {event.src_path}")
        
        # Wait for scanner to finish writing
        if self.wait_for_file_ready(event.src_path):
            try:
                self.process_and_split(event.src_path)
            except Exception as e:
                print(f"[!] Error processing {event.src_path}: {e}")

    def wait_for_file_ready(self, file_path, timeout=600):
        print("    Waiting for file write...", end="", flush=True)
        historical_size = -1
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                with open(file_path, 'rb'): pass # Check lock
                current_size = os.path.getsize(file_path)
                if current_size == historical_size and current_size > 0:
                    print(" Done.")
                    return True
                historical_size = current_size
                time.sleep(1)
            except: time.sleep(1)
        return False

    def get_text_start_end(self, page):
        """Extracts top 500 chars (header) and bottom 500 chars (footer)"""
        text = page.get_text("text")
        return text[:500], text[-500:]

    def is_new_document(self, curr_header, prev_footer):
        """Decides if the current page starts a new document."""
        # 1. FAST REGEX CHECKS
        
        # "Page 1 of X" - Strongest signal
        if re.search(r'(?i)page\s*1\s*(of|/|$)', curr_header): 
            return True, "Regex: Page 1 detected"
            
        # Previous page ended with "Page X of X" -> Next page must be new
        if prev_footer and re.search(r'(?i)page\s*(\d+)\s*of\s*\1', prev_footer):
            return True, "Regex: Previous page was final page"

        # 2. AI SEMANTIC CHECK
        # If regex fails, ask Llama: "Does this look like a new document?"
        prompt = f"""
        Review these two text snippets from consecutive pages of a scanned PDF pile.
        
        PREVIOUS PAGE BOTTOM (Footer context):
        "{prev_footer}"
        
        CURRENT PAGE TOP (Header context):
        "{curr_header}"
        
        TASK: Determine if 'CURRENT PAGE' is the START of a NEW document (e.g., new invoice, new letter) 
        or a CONTINUATION of the previous one.
        
        RULES:
        - If headers (dates, vendor names) change -> NEW.
        - If sentence cuts off -> CONTINUATION.
        - If page numbers reset -> NEW.
        
        Respond ONLY with JSON: {{"is_new": true/false, "confidence": 0-100}}
        """
        
        try:
            response = requests.post(API_URL, json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json",
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
            }, timeout=300)
            
            if response.status_code == 200:
                data = response.json()
                # Parse JSON string from 'response' field if needed, Ollama usually handles format:json well
                # But sometimes it puts the json in a string
                import json
                res_json = json.loads(data['response'])
                
                if res_json.get('is_new', False) and res_json.get('confidence', 0) > 70:
                    return True, "AI: High confidence split"
        except Exception as e:
            print(f"    [Warning] AI Check failed: {e}")
            
        return False, "Default: Continuation"

    def process_and_split(self, file_path):
        doc = fitz.open(file_path)
        total_pages = len(doc)
        print(f"    Analyzing {total_pages} pages for split points...")
        
        split_indices = [0] # Start of first doc
        
        # We need to track the *footer* of the previous page
        _, prev_footer = self.get_text_start_end(doc[0])
        
        for i in range(1, total_pages):
            curr_header, curr_footer = self.get_text_start_end(doc[i])
            
            is_split, reason = self.is_new_document(curr_header, prev_footer)
            
            if is_split:
                print(f"    [Split] Page {i+1}: {reason}")
                split_indices.append(i)
            
            prev_footer = curr_footer

        # Add the end of the file as the last "split" point
        split_indices.append(total_pages)
        
        # Save the chunks
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        print(f"    Splitting into {len(split_indices)-1} documents...")
        
        for k in range(len(split_indices) - 1):
            start = split_indices[k]
            end = split_indices[k+1]
            
            new_doc = fitz.open()
            new_doc.insert_pdf(doc, from_page=start, to_page=end-1)
            
            out_name = f"{base_name}_Part{k+1:02d}.pdf"
            out_path = os.path.join(OUTPUT_FOLDER, out_name)
            
            new_doc.save(out_path)
            print(f"    -> Saved: {out_name} (Pages {start+1}-{end})")
        
        doc.close()
        print("    [Done] Original file kept.")

if __name__ == "__main__":
    for folder in [WATCH_FOLDER, OUTPUT_FOLDER]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            
    observer = Observer()
    observer.schedule(PdfSplitHandler(), WATCH_FOLDER, recursive=False)
    
    print(f"--- AI DOCUMENT SPLITTER ACTIVE ---")
    print(f"Watching: {WATCH_FOLDER}")
    print(f"Saving to: {OUTPUT_FOLDER}")
    
    observer.start()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
