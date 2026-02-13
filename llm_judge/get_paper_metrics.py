#!/usr/bin/env python3
"""
Script to search for papers and retrieve detailed information including TNCSI and TNCSI_SP scores.

TNCSI (Topic-Normalized Citation Score Index):
- Measures how well a paper's citation count compares to similar papers in the same topic
- Uses GPT to extract a topic keyword from the paper's title and abstract
- Searches for similar papers using this keyword
- Fits an exponential distribution to citation counts of similar papers
- Calculates a normalized score (0-1) based on where the paper's citation count falls in this distribution

TNCSI_SP (TNCSI_S - Same Year):
- Similar to TNCSI but filters comparison papers to the same publication year
- Provides a more time-fair comparison by only comparing with papers published in the same year
- Uses the same topic keyword but with same_year=True filter

Usage:
    python get_paper_metrics.py --title "Your Paper Title"
    python get_paper_metrics.py --csv input.csv --output output.csv
    python get_paper_metrics.py --title "Paper Title" --abstract "Paper abstract..."
"""

import sys
import os
import argparse
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add the project root to the path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from PyBiblion.retrievers.semantic_scholar_paper import S2paper
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def save_api_data(paper, save_dir=None, paper_id=None):
    """
    Save full API response data for a paper to JSON file.
    
    Args:
        paper: S2paper object
        save_dir: Directory to save JSON files (if None, saves to 'paper_api_data' folder)
        paper_id: Optional identifier for the paper (used in filename)
        
    Returns:
        Path to saved JSON file or None if saving failed
    """
    if not paper.entity:
        return None
    
    try:
        # Create save directory
        if save_dir is None:
            save_dir = Path(BASE_DIR) / "paper_api_data/paper"
        else:
            save_dir = Path(save_dir)
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename from paper ID or title
        if paper_id:
            filename = f"{paper_id}_{paper.s2id or 'unknown'}.json"
        elif paper.s2id:
            filename = f"{paper.s2id}.json"
        else:
            # Use sanitized title
            safe_title = "".join(c for c in paper.title[:50] if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_title = safe_title.replace(' ', '_')
            filename = f"{safe_title}.json"
        
        filepath = save_dir / filename
        
        # Prepare data to save
        api_data = {
            'retrieved_at': datetime.now().isoformat(),
            'paper_id': paper_id,
            's2id': paper.s2id,
            'title': paper.title,
            'entity': paper.entity,  # Full API response
        }
        
        # Add additional data if available
        try:
            if hasattr(paper, 'authors') and paper.authors:
                api_data['authors'] = []
                for author in paper.authors:
                    # Access properties correctly - properties are s2_id, h_index, citation_count, paper_count
                    try:
                        author_data = {
                            'name': author.name,
                            'authorId': author.s2_id,  # Property name is s2_id
                            'hIndex': author.h_index,   # Property name is h_index
                            'citationCount': author.citation_count,  # Property name is citation_count
                            'paperCount': author.paper_count,  # Property name is paper_count
                            'affiliations': author.affiliations,  # Property name is affiliations
                        }
                    except Exception as e:
                        # Fallback if properties fail
                        logger.warning(f"Error accessing author properties: {e}")
                        author_data = {
                            'name': author.name,
                            'authorId': getattr(author, '_s2_id', None),
                            'hIndex': getattr(author, '_hIndex', None),
                            'citationCount': getattr(author, '_citationCount', None),
                            'paperCount': getattr(author, '_paperCount', None),
                            'affiliations': getattr(author, '_affiliations', None),
                        }
                    api_data['authors'].append(author_data)
        except Exception as e:
            logger.warning(f"Could not save authors data: {e}")
        
        # Save TNCSI results
        try:
            if paper._TNCSI:
                api_data['TNCSI_result'] = {
                    'TNCSI': paper._TNCSI.get('TNCSI'),
                    'topic': paper._TNCSI.get('topic'),
                    'loc': paper._TNCSI.get('loc'),
                    'scale': paper._TNCSI.get('scale'),
                }
        except:
            pass
        
        try:
            if paper._TNCSI_S:
                api_data['TNCSI_SP_result'] = {
                    'TNCSI': paper._TNCSI_S.get('TNCSI'),
                    'topic': paper._TNCSI_S.get('topic'),
                    'loc': paper._TNCSI_S.get('loc'),
                    'scale': paper._TNCSI_S.get('scale'),
                }
        except:
            pass
        
        # Save to JSON file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(api_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Saved API data to: {filepath}")
        return str(filepath)
        
    except Exception as e:
        logger.error(f"Error saving API data: {e}")
        return None


def get_paper_details(paper_title, abstract=None, force_return=True, save_api=False, save_dir=None, paper_id=None):
    """
    Search for a paper and retrieve detailed information including TNCSI and TNCSI_SP scores.
    
    API CALL SUMMARY (per paper, first run, no caching):
    - OpenAI API: 1 call (GPT keyword extraction)
    - Semantic Scholar API: ~12-20+ calls
      * 1 call: Paper search by title
      * 0-1 call: Authors (if filled_authors=True)
      * ~10 calls: TNCSI (search similar papers, 100 per call, up to 1000 papers)
      * ~10 calls: TNCSI_SP (search similar papers same year, 100 per call, up to 1000 papers)
      * 2-10+ calls: IEI (fetch citations, 1000 per call, depends on citation count) [optional]
      * 1 call: RQM (fetch references) [optional]
    
    With caching enabled (default): Most calls cached after first run.
    See API_CALL_ANALYSIS.md for detailed breakdown.
    
    Args:
        paper_title: Title of the paper to search for
        abstract: Optional abstract to help with keyword extraction
        force_return: If True, return results even if title doesn't match exactly
        save_api: If True, save full API response to JSON file
        save_dir: Directory to save API data (if None, uses default 'paper_api_data')
        paper_id: Optional identifier for the paper (used in filename)
        
    Returns:
        Dictionary containing paper details and metrics
    """
    try:
        logger.info(f"Searching for paper: {paper_title}")
        
        # Create S2paper object
        paper = S2paper(paper_title, ref_type='title', force_return=force_return, use_cache=True)
      
        # Check if paper was found
        if not paper.entity:
            logger.warning(f"Paper not found: {paper_title}")
            return {
                'title': paper_title,
                'found': False,
                'error': 'Paper not found in Semantic Scholar'
            }
        
        # Get basic information
        result = {
            'title': paper.title,
            's2id': paper.s2id,
            'found': True,
            'abstract': paper.abstract,
            'publication_date': str(paper.publication_date) if paper.publication_date else None,
            'citation_count': paper.citation_count,
            'reference_count': paper.reference_count,
            'influential_citation_count': paper.influential_citation_count,
            'venue': paper.publication_source,
            'publisher': paper.publisher,
            'doi': paper.DOI,
            'field': paper.field,
            'tldr': paper.tldr,
        }
        
        # Get authors information
        try:
            authors = paper.authors
            if authors:
                result['authors'] = [author.name for author in authors]
                result['author_count'] = len(authors)
            else:
                result['authors'] = []
                result['author_count'] = 0
        except Exception as e:
            logger.warning(f"Could not retrieve authors: {e}")
            result['authors'] = []
            result['author_count'] = 0
        
        # Get GPT keyword (used for TNCSI calculation)
        try:
            logger.info("Extracting topic keyword using GPT...")
            # result['gpt_keyword'] = paper.gpt_keyword
            result['gpt_keyword'] = paper.openrouter_keyword

        except Exception as e:
            logger.warning(f"Could not extract GPT keyword: {e}")
            result['gpt_keyword'] = None
        
        # Get TNCSI score
        try:
            logger.info("Calculating TNCSI score...")
            tncsi_result = paper.TNCSI
            if tncsi_result:
                result['TNCSI'] = tncsi_result.get('TNCSI', -1)
                result['TNCSI_topic'] = tncsi_result.get('topic', 'NONE')
                result['TNCSI_loc'] = tncsi_result.get('loc')
                result['TNCSI_scale'] = tncsi_result.get('scale')
            else:
                result['TNCSI'] = -1
                result['TNCSI_topic'] = 'NONE'
        except Exception as e:
            logger.error(f"Error calculating TNCSI: {e}")
            result['TNCSI'] = -1
            result['TNCSI_topic'] = 'ERROR'
        
        # Get TNCSI_SP (TNCSI_S) score - Same Year version
        # Note: Property is called TNCSI_S but represents TNCSI_SP (Same Year)
        try:
            logger.info("Calculating TNCSI_SP (same year) score...")
            tncsi_s_result = paper.TNCSI_S  # Property name is TNCSI_S, but it's TNCSI_SP (Same Year)
            if tncsi_s_result:
                result['TNCSI_SP'] = tncsi_s_result.get('TNCSI', -1)
                result['TNCSI_SP_topic'] = tncsi_s_result.get('topic', 'NONE')
                result['TNCSI_SP_loc'] = tncsi_s_result.get('loc')
                result['TNCSI_SP_scale'] = tncsi_s_result.get('scale')
            else:
                result['TNCSI_SP'] = -1
                result['TNCSI_SP_topic'] = 'NONE'
        except Exception as e:
            logger.error(f"Error calculating TNCSI_SP: {e}")
            result['TNCSI_SP'] = -1
            result['TNCSI_SP_topic'] = 'ERROR'
        
        # Save API data if requested
        if save_api:
            api_filepath = save_api_data(paper, save_dir=save_dir, paper_id=paper_id)
            result['api_data_file'] = api_filepath
        
        # # Get other metrics if available
        # try:
        #     if paper.publication_date and paper.citation_count:
        #         iei_result = paper.IEI
        #         result['IEI_L6'] = iei_result.get('L6', float('-inf'))
        #         result['IEI_I6'] = iei_result.get('I6', float('-inf'))
        # except Exception as e:
        #     logger.warning(f"Could not calculate IEI: {e}")
        #     result['IEI_L6'] = None
        #     result['IEI_I6'] = None
        
        # try:
        #     if paper.publication_date and paper.reference_count:
        #         rqm_result = paper.RQM
        #         result['RQM'] = rqm_result.get('RQM')
        #         result['RQM_ARQ'] = rqm_result.get('ARQ')
        # except Exception as e:
        #     logger.warning(f"Could not calculate RQM: {e}")
        #     result['RQM'] = None
        #     result['RQM_ARQ'] = None
        
        logger.info(f"Successfully retrieved information for: {paper.title}")
        return result
        
    except Exception as e:
        logger.error(f"Error processing paper '{paper_title}': {e}")
        import traceback
        traceback.print_exc()
        return {
            'title': paper_title,
            'found': False,
            'error': str(e)
        }


def process_csv(input_csv, output_csv=None, title_column='title', abstract_column='abstract', 
                save_api=False, save_dir=None, id_column=None):
    """
    Process a CSV file with paper titles and get metrics for each paper.
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file (if None, adds '_with_metrics' to input filename)
        title_column: Name of the column containing paper titles
        abstract_column: Name of the columncompute_bm25_score containing abstracts (optional)
        save_api: If True, save full API response for each paper to JSON
        save_dir: Directory to save API data files
        id_column: Name of column to use as paper ID (for JSON filenames)
    """
    logger.info(f"Reading CSV file: {input_csv}")
    df = pd.read_csv(input_csv)
    
    if title_column not in df.columns:
        raise ValueError(f"Column '{title_column}' not found in CSV. Available columns: {df.columns.tolist()}")
    
    results = []
    total = len(df)

    for idx, row in tqdm(df.iterrows()):
        logger.info(f"Processing paper {idx + 1}/{total}")
        if idx < 111:
            continue
        paper_title = str(row[title_column])
        abstract = str(row[abstract_column]) if abstract_column in df.columns and pd.notna(row.get(abstract_column)) else None

        # Get paper ID if available
        paper_id = None
        process_entry = True  # For id_col control


        paper_id = str(row['id'])
        process_entry = True

        if process_entry:
            result = get_paper_details(
                paper_title, 
                abstract=abstract, 
                save_api=save_api,
                save_dir=save_dir,
                paper_id=paper_id
            )

        results.append(result)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Merge with original data
    output_df = pd.concat([df, results_df], axis=1)
    
    # Save to output file
    if output_csv is None:
        input_path = Path(input_csv)
        output_csv = input_path.parent / f"{input_path.stem}_with_metrics{input_path.suffix}"
    
    logger.info(f"Saving results to: {output_csv}")
    output_df.to_csv(output_csv, index=False)
    logger.info(f"Saved {len(output_df)} papers with metrics to {output_csv}")
    
    return output_df


def print_paper_details(result):
    """Pretty print paper details."""
    print("\n" + "="*80)
    print("PAPER DETAILS")
    print("="*80)
    
    if not result.get('found', False):
        print(f"Title: {result.get('title', 'N/A')}")
        print(f"Status: NOT FOUND")
        if 'error' in result:
            print(f"Error: {result['error']}")
        return
    
    print(f"Title: {result.get('title', 'N/A')}")
    print(f"Semantic Scholar ID: {result.get('s2id', 'N/A')}")
    print(f"Publication Date: {result.get('publication_date', 'N/A')}")
    print(f"Venue: {result.get('venue', 'N/A')}")
    print(f"Publisher: {result.get('publisher', 'N/A')}")
    print(f"DOI: {result.get('doi', 'N/A')}")
    print(f"Field: {result.get('field', 'N/A')}")
    print(f"\nCitation Count: {result.get('citation_count', 0)}")
    print(f"Influential Citations: {result.get('influential_citation_count', 0)}")
    print(f"Reference Count: {result.get('reference_count', 0)}")
    
    if result.get('authors'):
        print(f"\nAuthors ({result.get('author_count', 0)}):")
        for author in result['authors'][:5]:  # Show first 5
            print(f"  - {author}")
        if result.get('author_count', 0) > 5:
            print(f"  ... and {result.get('author_count', 0) - 5} more")
    
    print(f"\nTopic Keyword (GPT): {result.get('gpt_keyword', 'N/A')}")
    
    print("\n" + "-"*80)
    print("METRICS")
    print("-"*80)
    
    tncsi = result.get('TNCSI', -1)
    if tncsi != -1:
        print(f"TNCSI Score: {tncsi:.4f}")
        print(f"  Topic: {result.get('TNCSI_topic', 'N/A')}")
    else:
        print("TNCSI Score: Not available")
    
    tncsi_sp = result.get('TNCSI_SP', -1)
    if tncsi_sp != -1:
        print(f"TNCSI_SP Score (Same Year): {tncsi_sp:.4f}")
        print(f"  Topic: {result.get('TNCSI_SP_topic', 'N/A')}")
    else:
        print("TNCSI_SP Score: Not available")
    
    if result.get('IEI_L6') is not None:
        print(f"\nIEI Metrics:")
        print(f"  L6: {result.get('IEI_L6', 'N/A')}")
        print(f"  I6: {result.get('IEI_I6', 'N/A')}")
    
    if result.get('RQM') is not None:
        print(f"\nRQM Metrics:")
        print(f"  RQM: {result.get('RQM', 'N/A')}")
        print(f"  ARQ: {result.get('RQM_ARQ', 'N/A')}")
    
    if result.get('abstract'):
        print(f"\nAbstract (first 200 chars):")
        print(f"  {result.get('abstract', '')[:200]}...")
    
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Search for papers and retrieve TNCSI and TNCSI_SP scores',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Search for a single paper by title
  python get_paper_metrics.py --title "Attention Is All You Need"
  
  # Search with abstract
  python get_paper_metrics.py --title "Your Paper" --abstract "Paper abstract here..."
  
  # Process CSV file
  python get_paper_metrics.py --csv papers.csv --output results.csv
  
  # Process CSV with custom column names
  python get_paper_metrics.py --csv papers.csv --title-col "paper_title" --abstract-col "paper_abstract"
  
  # Save API data to JSON files
  python get_paper_metrics.py --csv papers.csv --save-api --save-dir paper_data --id-col id

  python get_paper_metrics_naidv1.py --csv /mnt/data/son/Reviewerly/NAIP/v1_resource/NAIDv1/NAID_test_extrainfo.csv --save-api --save-dir /mnt/data/son/Reviewerly/paper_api_data_naidv1_test_deepseek/paper --title-col "title"  --abstract-col "abstract" 

  python get_paper_metrics_naidv1.py --csv /mnt/data/son/Reviewerly/NAIP/v2_resource/NAIDv2/NAIDv2-test.csv --save-api --save-dir /mnt/data/son/Reviewerly/paper_api_data_naidv2_test_llama/paper --title-col "title"  --abstract-col "abstract" 

        """
    )
    
    parser.add_argument('--title', type=str, help='Paper title to search for')
    parser.add_argument('--abstract', type=str, help='Paper abstract (optional, helps with keyword extraction)')
    parser.add_argument('--csv', type=str, help='Input CSV file with paper titles')
    parser.add_argument('--output', type=str, help='Output CSV file (for CSV processing mode)')
    parser.add_argument('--title-col', type=str, default='title', 
                       help='Name of title column in CSV (default: title)')
    parser.add_argument('--abstract-col', type=str, default='abstract',
                       help='Name of abstract column in CSV (default: abstract)')
    parser.add_argument('--id-col', type=str, default=None,
                       help='Name of ID column in CSV (used for JSON filenames)')
    parser.add_argument('--force-return', action='store_true', default=True,
                       help='Return results even if title doesn\'t match exactly')
    parser.add_argument('--save-api', action='store_true',
                       help='Save full API response data to JSON files for each paper')
    parser.add_argument('--save-dir', type=str, default=None,
                       help='Directory to save API data JSON files (default: paper_api_data/)')
    
    args = parser.parse_args()
    
    if args.csv:
        # Process CSV file
        process_csv(
            args.csv, 
            args.output, 
            args.title_col, 
            args.abstract_col,
            save_api=args.save_api,
            save_dir=args.save_dir,
            id_column=args.id_col
        )
    elif args.title:
        # Process single paper
        result = get_paper_details(
            args.title, 
            abstract=args.abstract, 
            force_return=args.force_return,
            save_api=args.save_api,
            save_dir=args.save_dir
        )
        print_paper_details(result)
        if args.save_api and result.get('api_data_file'):
            print(f"\n💾 API data saved to: {result['api_data_file']}")
    else:
        parser.print_help()
        print("\nError: Either --title or --csv must be provided.")
        sys.exit(1)


if __name__ == '__main__':
    main()
