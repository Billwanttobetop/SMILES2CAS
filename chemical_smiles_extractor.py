import pandas as pd
import requests
import threading
import time
import os
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from rdkit import Chem
from rdkit.Chem import AllChem
import json
import re
import pickle
try:
    import pubchempy as pcp
    PUBCHEMPY_AVAILABLE = True
except ImportError:
    print("提示: pubchempy库未安装，将不使用子结构搜索功能。可通过'pip install pubchempy'安装。")
    PUBCHEMPY_AVAILABLE = False

def validate_smiles(smiles):
    """验证SMILES是否有效，返回是否有效及标准化SMILES"""
    if not smiles or not isinstance(smiles, str):
        return False, None
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, None
        
        # 获取标准化的SMILES
        canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        return True, canonical_smiles
    except Exception as e:
        print(f"SMILES验证错误 '{smiles}': {str(e)}")
        return False, None

class CASRetriever:
    def __init__(self, max_retries=3, retry_delay=1, cache_file="cas_cache.pkl"):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Initialize cache
        self.cache_file = cache_file
        self.cache = self._load_cache()
        
        # Store statistics
        self.stats = {
            "pubchem_hits": 0,
            "pubchempy_hits": 0,
            "chemspider_hits": 0,
            "cactus_hits": 0,
            "molport_hits": 0,
            "commonchemistry_hits": 0,
            "cache_hits": 0,
            "total_queries": 0,
            "invalid_smiles": 0
        }

    def _load_cache(self):
        """Load cache from file if exists."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading cache: {str(e)}")
        return {}

    def _save_cache(self):
        """Save cache to file."""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            print(f"Error saving cache: {str(e)}")

    def get_cas_from_pubchem(self, smiles):
        """Get CAS number from PubChem using SMILES by first fetching CID."""
        cid = None
        try:
            # Step 1: Get CID from SMILES
            for attempt in range(self.max_retries):
                try:
                    url_cid = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/cids/JSON"
                    response_cid = self.session.get(url_cid, timeout=10)
                    
                    if response_cid.status_code == 200:
                        data_cid = response_cid.json()
                        if 'IdentifierList' in data_cid and 'CID' in data_cid['IdentifierList']:
                            cid = data_cid['IdentifierList']['CID'][0]
                            break  # CID found
                    elif response_cid.status_code != 429: # Not a rate limit error
                        break 
                except requests.RequestException:
                    pass # Handled by retry
                except json.JSONDecodeError: # Invalid JSON response
                    pass # Handled by retry
                time.sleep(self.retry_delay * (attempt + 1))
            
            if not cid:
                # print(f"PubChem: Could not get CID for {smiles}")
                return None

            # Step 2: Get CAS from CID
            for attempt in range(self.max_retries):
                try:
                    url_cas = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CASNumber/JSON"
                    response_cas = self.session.get(url_cas, timeout=10)

                    if response_cas.status_code == 200:
                        data_cas = response_cas.json()
                        properties = data_cas.get('PropertyTable', {}).get('Properties', [{}])
                        if properties and 'CASNumber' in properties[0]:
                            cas = properties[0]['CASNumber']
                            # Validate CAS format (numbers-numbers-number)
                            if isinstance(cas, str) and '-' in cas:
                                parts = cas.split('-')
                                if len(parts) == 3 and all(p.isdigit() for p in parts):
                                    self.stats["pubchem_hits"] += 1
                                    return cas
                            # Sometimes CAS might be in a list
                            elif isinstance(cas, list) and cas:
                                for c_item in cas:
                                     if isinstance(c_item, str) and '-' in c_item:
                                        parts = c_item.split('-')
                                        if len(parts) == 3 and all(p.isdigit() for p in parts):
                                            self.stats["pubchem_hits"] += 1
                                            return c_item # Return the first valid CAS
                        break # Processed response, break retry loop for CAS
                    elif response_cas.status_code != 429: # Not a rate limit error
                        break
                except requests.RequestException:
                    pass
                except json.JSONDecodeError:
                    pass
                time.sleep(self.retry_delay * (attempt + 1))

        except Exception as e:
            print(f"PubChem error for {smiles} (CID: {cid}): {str(e)}")
        
        return None

    def get_cas_from_pubchempy(self, smiles):
        """Get CAS number using PubChemPy library with substructure search."""
        if not PUBCHEMPY_AVAILABLE:
            return None
            
        try:
            cas_results = []
            for attempt in range(self.max_retries):
                try:
                    # 首先尝试精确搜索
                    results = pcp.get_synonyms(smiles, 'smiles')
                    for result in results:
                        for syn in result.get('Synonym', []):
                            match = re.match(r'^\d{2,7}-\d\d-\d$', syn)
                            if match:
                                cas_results.append(syn)
                    
                    # 如果精确搜索没找到，尝试子结构搜索
                    if not cas_results:
                        results = pcp.get_synonyms(smiles, 'smiles', searchtype='substructure', listkey_count=5)
                        for result in results:
                            for syn in result.get('Synonym', []):
                                match = re.match(r'^\d{2,7}-\d\d-\d$', syn)
                                if match:
                                    cas_results.append(syn)
                                    
                    # 如果找到了CAS号，返回第一个
                    if cas_results:
                        self.stats["pubchempy_hits"] += 1
                        return cas_results[0]
                    
                    break  # 如果执行到这里，说明没有发生异常，可以退出尝试循环
                
                except pcp.PubChemHTTPError as e:
                    if "too many requests" in str(e).lower() or "429" in str(e):
                        time.sleep(self.retry_delay * (attempt + 1))
                    else:
                        break  # 其他HTTP错误，停止尝试
                
                except Exception as e:
                    print(f"PubChemPy error for {smiles}: {str(e)}")
                    break  # 其他错误，停止尝试
            
        except Exception as e:
            print(f"PubChemPy general error for {smiles}: {str(e)}")
        
        return None

    def get_cas_from_chemspider(self, smiles):
        """Get CAS number from ChemSpider using SMILES."""
        try:
            # Convert SMILES to mol and then to InChI Key for more reliable search
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                inchikey = Chem.MolToInchiKey(mol)
                
                for attempt in range(self.max_retries):
                    try:
                        url = f"https://www.chemspider.com/InchiKey/{inchikey}"
                        response = self.session.get(url, timeout=10)
                        
                        if response.status_code == 200:
                            # Simple text extraction - look for CAS pattern
                            text = response.text
                            cas_pattern = r"CAS:\s*(\d+-\d+-\d+)"
                            match = re.search(cas_pattern, text)
                            if match:
                                self.stats["chemspider_hits"] += 1
                                return match.group(1)
                        elif response.status_code != 429:  # If not rate limited
                            break
                    except requests.RequestException:
                        pass
                    
                    time.sleep(self.retry_delay * (attempt + 1))
                    
        except Exception as e:
            print(f"ChemSpider error for {smiles}: {str(e)}")
        
        return None

    def get_cas_from_cactus(self, smiles):
        """Get CAS number from NCI/CADD Chemical Identifier Resolver using SMILES."""
        try:
            for attempt in range(self.max_retries):
                try:
                    url = f"https://cactus.nci.nih.gov/chemical/structure/{smiles}/cas"
                    response = self.session.get(url, timeout=10)
                    
                    if response.status_code == 200:
                        cas = response.text.strip()
                        # Validate CAS format (numbers-numbers-number)
                        if '-' in cas:
                            parts = cas.split('-')
                            if len(parts) == 3 and all(p.isdigit() for p in parts):
                                self.stats["cactus_hits"] += 1
                                return cas
                    elif response.status_code != 429:  # If not rate limited
                        break
                except requests.RequestException:
                    pass
                
                time.sleep(self.retry_delay * (attempt + 1))
                
        except Exception as e:
            print(f"Cactus error for {smiles}: {str(e)}")
        
        return None

    def get_cas_from_molport(self, smiles):
        """Get CAS number from MolPort using SMILES or InchiKey."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                inchikey = Chem.MolToInchiKey(mol)
                
                for attempt in range(self.max_retries):
                    try:
                        url = f"https://www.molport.com/shop/moleculelink/molecule-search?searchType=text&searchQuery={inchikey}"
                        response = self.session.get(url, timeout=10)
                        
                        if response.status_code == 200:
                            text = response.text
                            # Look for CAS pattern in the response
                            cas_pattern = r"CAS:\s*(\d+-\d+-\d+)"
                            match = re.search(cas_pattern, text)
                            if match:
                                self.stats["molport_hits"] += 1
                                return match.group(1)
                        elif response.status_code != 429:
                            break
                    except requests.RequestException:
                        pass
                    
                    time.sleep(self.retry_delay * (attempt + 1))
        except Exception as e:
            print(f"MolPort error for {smiles}: {str(e)}")
        
        return None

    def get_cas_from_commonchemistry(self, smiles):
        """Get CAS number from Common Chemistry using InChI Key."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                inchikey = Chem.MolToInchiKey(mol)
                first_part = inchikey.split('-')[0] if '-' in inchikey else inchikey
                
                for attempt in range(self.max_retries):
                    try:
                        url = f"https://commonchemistry.cas.org/api/search?q={first_part}"
                        response = self.session.get(url, timeout=10)
                        
                        if response.status_code == 200:
                            data = response.json()
                            results = data.get('results', [])
                            if results and 'rn' in results[0]:
                                cas = results[0]['rn']
                                # Validate CAS format
                                if '-' in cas:
                                    parts = cas.split('-')
                                    if len(parts) == 3 and all(p.isdigit() for p in parts):
                                        self.stats["commonchemistry_hits"] += 1
                                        return cas
                        elif response.status_code != 429:
                            break
                    except requests.RequestException:
                        pass
                    except json.JSONDecodeError:
                        pass
                    
                    time.sleep(self.retry_delay * (attempt + 1))
        except Exception as e:
            print(f"Common Chemistry error for {smiles}: {str(e)}")
        
        return None
    
    def get_cas_from_all_sources(self, smiles):
        """Try multiple sources to get CAS number using cache."""
        self.stats["total_queries"] += 1
        
        # Validate SMILES
        is_valid, canonical_smiles = validate_smiles(smiles)
        if not is_valid:
            self.stats["invalid_smiles"] += 1
            print(f"Invalid SMILES: {smiles}")
            return None
            
        # Use canonical SMILES for consistency
        smiles = canonical_smiles
        
        # Check cache first
        if smiles in self.cache:
            self.stats["cache_hits"] += 1
            return self.cache[smiles]
        
        # Try each source in order
        cas = self.get_cas_from_pubchem(smiles)
        if cas:
            self.cache[smiles] = cas
            return cas
            
        cas = self.get_cas_from_pubchempy(smiles)
        if cas:
            self.cache[smiles] = cas
            return cas
        
        cas = self.get_cas_from_cactus(smiles)
        if cas:
            self.cache[smiles] = cas
            return cas
        
        cas = self.get_cas_from_chemspider(smiles)
        if cas:
            self.cache[smiles] = cas
            return cas
        
        cas = self.get_cas_from_commonchemistry(smiles)
        if cas:
            self.cache[smiles] = cas
            return cas
            
        cas = self.get_cas_from_molport(smiles)
        if cas:
            self.cache[smiles] = cas
            return cas
        
        # No CAS found in any source
        self.cache[smiles] = None
        return None

    def print_stats(self):
        """Print statistics about CAS retrievals."""
        print("\n--- CAS Retrieval Statistics ---")
        print(f"Total queries: {self.stats['total_queries']}")
        print(f"Invalid SMILES: {self.stats['invalid_smiles']}")
        print(f"Cache hits: {self.stats['cache_hits']} ({self.stats['cache_hits']/max(1, self.stats['total_queries'])*100:.1f}%)")
        print(f"PubChem hits: {self.stats['pubchem_hits']}")
        print(f"PubChemPy hits: {self.stats['pubchempy_hits']}")
        print(f"Cactus hits: {self.stats['cactus_hits']}")
        print(f"ChemSpider hits: {self.stats['chemspider_hits']}")
        print(f"Common Chemistry hits: {self.stats['commonchemistry_hits']}")
        print(f"MolPort hits: {self.stats['molport_hits']}")
        print("-----------------------------")

def smiles_worker(cas_retriever, smiles_queue, processed_smiles_cache):
    """Worker function to process individual SMILES in a thread and populate a shared cache."""
    while not smiles_queue.empty():
        try:
            original_smiles, canonical_smiles = smiles_queue.get()
            
            # The CAS retriever handles its own internal cache (disk/memory)
            # and also updates it if a new CAS is found.
            cas_number = cas_retriever.get_cas_from_all_sources(canonical_smiles)
            
            # Store in a temporary in-memory cache for this run for quick lookup later
            # The CASRetriever's cache is the primary persistent one.
            processed_smiles_cache[canonical_smiles] = cas_number
            
            print(f"Processed SMILES: {canonical_smiles} → {cas_number if cas_number else 'Not found'}")
            
            # Periodically save the CASRetriever's main cache
            # (This might be too frequent if many short tasks, consider adjusting)
            # For now, let CASRetriever handle its own saving logic if any, or save at end.

        except Exception as e:
            print(f"Error in smiles_worker for {canonical_smiles}: {str(e)}")
        finally:
            smiles_queue.task_done()

def main():
    # 配置参数 - 可以根据需要修改这些参数
    input_file = "Pubchem_P_ourmodel_v1_lowest5000.csv"  # 输入文件名
    smiles_column = "SMILES"  # 要处理的SMILES列名，可以根据实际情况修改
    output_suffix = "_with_cas_v2"  # 输出文件后缀
    
    # Check if file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # 检查指定的SMILES列是否存在
    if smiles_column not in df.columns:
        print(f"Error: Column '{smiles_column}' not found in the input file.")
        print(f"Available columns: {list(df.columns)}")
        return
    
    # --- Step 1: Validate SMILES and collect unique canonical SMILES --- 
    all_unique_canonical_smiles = set()
    smiles_validation_map = {} # Store original -> canonical mapping for later use
    invalid_smiles_records = []

    print(f"Validating SMILES strings from column '{smiles_column}'...")
    for idx, row in df.iterrows():
        original_smiles = row[smiles_column]
        if pd.isna(original_smiles) or not isinstance(original_smiles, str) or not original_smiles.strip():
            # Handle empty or non-string SMILES as invalid implicitly
            invalid_smiles_records.append((idx, smiles_column, original_smiles, "Empty or invalid type"))
            continue

        is_valid, canonical_smiles = validate_smiles(original_smiles)
        if is_valid:
            all_unique_canonical_smiles.add(canonical_smiles)
            smiles_validation_map[original_smiles] = canonical_smiles
        else:
            invalid_smiles_records.append((idx, smiles_column, original_smiles, "RDKit validation failed"))

    # Report invalid SMILES
    if invalid_smiles_records:
        print(f"发现 {len(invalid_smiles_records)} 个无效或格式不正确的SMILES条目:")
        # Deduplicate invalid records for reporting by original_smiles
        reported_invalids = set()
        for i, (idx, col_name, smiles_val, reason) in enumerate(invalid_smiles_records):
            if smiles_val not in reported_invalids:
                if i < 10: # Show first 10 unique invalid entries
                    print(f"  行 {idx+2}, SMILES: '{smiles_val}', 原因: {reason}")
                reported_invalids.add(smiles_val)
        if len(reported_invalids) > 10:
            print(f"  ...以及 {len(reported_invalids) - 10} 个更多独特的无效条目")
            
        invalid_df_data = []
        for idx, col_name, smiles_val, reason in invalid_smiles_records:
            invalid_df_data.append({"行号(原始CSV)": idx + 2, "列名": col_name, "SMILES": smiles_val, "错误原因": reason})
        
        invalid_df = pd.DataFrame(invalid_df_data)
        # Deduplicate before saving to file, based on SMILES
        invalid_df.drop_duplicates(subset=["SMILES"], keep='first', inplace=True)
        invalid_df.to_csv("无效SMILES详情.csv", index=False, encoding='utf-8-sig')
        print(f"无效SMILES的详细信息已保存至 '无效SMILES详情.csv'")

    # --- Step 2: Populate queue with unique canonical SMILES for CAS retrieval --- 
    smiles_processing_queue = Queue()
    for s in all_unique_canonical_smiles:
        # The CASRetriever will use its cache, so we pass original for context if needed, but canonical for processing
        smiles_processing_queue.put(("N/A_original", s)) # Original SMILES not strictly needed by worker here

    print(f"\n总共发现 {len(all_unique_canonical_smiles)} 个独特的、有效的、规范化的SMILES进行CAS号查询。")
    if not all_unique_canonical_smiles:
        print("没有有效的SMILES可供处理。退出程序。")
        return

    # --- Step 3: Process unique SMILES in parallel --- 
    # This cache is temporary for this run, CASRetriever has its own persistent cache
    processed_smiles_cas_cache = {} 
    cas_retriever = CASRetriever(cache_file="cas_cache.pkl")
    
    num_threads = 5
    threads = []
    print(f"启动 {num_threads} 个线程处理SMILES查询...")
    for _ in range(num_threads):
        thread = threading.Thread(target=smiles_worker, args=(cas_retriever, smiles_processing_queue, processed_smiles_cas_cache))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to finish
    smiles_processing_queue.join() # Wait for queue to be empty
    for thread in threads:
        thread.join() # Wait for threads to terminate
    
    print("所有SMILES处理完毕。")
    # Save final CASRetriever cache (contains all lookups from this run and previous)
    cas_retriever._save_cache()

    # --- Step 4: Populate DataFrame with CAS numbers and canonical SMILES --- 
    df[f'{smiles_column}_cas'] = ''
    df[f'{smiles_column}_canonical'] = ''

    print("正在将CAS号和规范SMILES映射回原始数据表...")
    for idx, row in df.iterrows():
        original_smiles = row[smiles_column]
        
        canonical_form = smiles_validation_map.get(original_smiles)

        if canonical_form:
            df.at[idx, f'{smiles_column}_canonical'] = canonical_form
            # Use CASRetriever's cache as the single source of truth for CAS numbers
            df.at[idx, f'{smiles_column}_cas'] = cas_retriever.cache.get(canonical_form, '') 
        else:
            df.at[idx, f'{smiles_column}_cas'] = 'INVALID_SMILES' # Mark if original was invalid
            
    # --- Step 5: Save results and report --- 
    base_filename = os.path.splitext(input_file)[0]
    output_file = f"{base_filename}{output_suffix}.csv"
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    # Create dataframe with SMILES that couldn't be matched (for valid canonical SMILES)
    not_found_data = []
    for smiles_str in all_unique_canonical_smiles:
        if not cas_retriever.cache.get(smiles_str):
            not_found_data.append({'canonical_smiles': smiles_str, 'cas_number': 'Not Found'})
    
    if not_found_data:
        not_found_df = pd.DataFrame(not_found_data)
        not_found_file = "未匹配CAS号的规范SMILES.csv"
        not_found_df.to_csv(not_found_file, index=False, encoding='utf-8-sig')
        print(f"未找到CAS号的规范SMILES列表已保存至 '{not_found_file}'")
    else:
        print("所有有效的规范SMILES都已成功匹配或确定无CAS号(已缓存)。")

    cas_retriever.print_stats()
    
    print(f"\n处理完成!")
    print(f"结果已保存至 '{output_file}'")
    
    # Calculate success rate for valid canonical SMILES that were processed
    successfully_matched_count = 0
    for s in all_unique_canonical_smiles:
        if cas_retriever.cache.get(s):
            successfully_matched_count += 1
            
    total_valid_unique_smiles = len(all_unique_canonical_smiles)
    print(f"处理的独特有效规范SMILES数量: {total_valid_unique_smiles}")
    print(f"成功匹配到CAS号的独特SMILES数量: {successfully_matched_count}")
    if total_valid_unique_smiles > 0:
        print(f"独特SMILES的CAS匹配成功率: {successfully_matched_count/total_valid_unique_smiles*100:.1f}%")

if __name__ == '__main__':
    main() 