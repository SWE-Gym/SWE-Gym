import json
import os
from moatless.workspace import Workspace

code_file_extensions = ["1.ada", ".2.ada", ".ada", ".adb", ".ads", ".asm", ".asp", ".aspx", ".bas", ".bash", ".bat", ".c++", ".c", ".cbl", ".cc", ".class", ".clj", ".cob", ".cpp", ".cs", ".csh", ".cxx", ".d", ".diff", ".dll", ".e", ".el", ".f", ".f77", ".f90", ".fish", ".for", ".fth", ".ftn", ".go", ".groovy", ".h", ".hh", ".hpp", ".hs", ".htm", ".html", ".hxx", ".inc", ".java", ".js", ".json", ".jsp", ".jsx", ".ksh", ".kt", ".kts", ".lhs", ".lisp", ".lua", ".m", ".m4", ".nim", ".patch", ".php", ".php3", ".php4", ".php5", ".phtml", ".pl", ".po", ".pp", ".prql", ".py", ".ps1", ".psd1", ".psm1", ".ps1xml", ".psc1", ".pssc", ".psrc", ".r", ".rb", ".rs", ".s", ".scala", ".sh", ".sql", ".swg", ".swift", ".v", ".vb", ".vcxproj", ".wll", ".xcodeproj", ".xml", ".xll", ".zig", ".zsh"]

def extract_spans_from_trajectory(trajectory_path, instance, repo_base_dir):
    """Extract spans from a trajectory file for a given instance ID.
    
    Args:
        trajectory_path (str): Path to trajectory directory
        instance_id (str): Instance ID to extract spans for
        
    Returns:
        list: List of identified spans with content
    """
    instance_id = instance["instance_id"]
    identified_spans = get_identified_spans(instance, trajectory_path)
    
    repo_dir = setup_swebench_repo(instance, repo_base_dir)
    workspace = Workspace.from_dirs(repo_dir=repo_dir, index_dir=None)
    
    for identified_span in identified_spans:
        file_path = identified_span["file_path"]
        file = workspace.file_repo.get_file(file_path)
        
        with open(os.path.join(repo_dir, file_path)) as f:
            content_lines = f.readlines()
        
        identified_span["content"] = []
        
        for span_id in identified_span["span_ids"]:
            span = file.module.find_span_by_id(span_id)
            if not span:
                print(f"Span not found: {span_id} in {file_path}, Instance {instance_id}")
                continue
            
            start_line = span.start_line
            end_line = span.end_line
            
            content = content_lines[start_line - 1: end_line]
            identified_span["content"].append({
                "span_id": span_id,
                "start_line": start_line,
                "end_line": end_line,
                "content": content
            })
            
    return identified_spans

def merge_spans(identified_spans):
    # we first store them as dict, key is file_path + span_ids,
    # then decode
    spans_dict = {}
    for this_exp_spans in identified_spans:
        for span in this_exp_spans:
            spans_dict[span["file_path"] + "-" + str(span["span_ids"])] = span
    merged_spans = list(spans_dict.values())
    return merged_spans

def organize_identified_spans(identified_spans):
    concat_spans = ""
    for identified_span in identified_spans:
        file_path = identified_span["file_path"]
        for span in identified_span["content"]:
            content = span["content"]
            span_id = span["span_id"]
            start_line = span["start_line"]
            # pad the line numbers to 4 digits
            line_numbered_content = ''.join([f"({start_line + i:04d}) {line}" for i, line in enumerate(content)])
            wrapped_code_span = f"<code_span={file_path} span_id={span_id}>\n{line_numbered_content}\n</code_span>"
            if not concat_spans:
                concat_spans = wrapped_code_span
            else:
                concat_spans += f"\n\n{wrapped_code_span}"
    return concat_spans

def extract_gpt4_tag(text, tag):
    if f"<{tag}>" not in text:
        return None
    if f"</{tag}>" not in text.split(f"<{tag}>")[-1]:
        return None
    return text.split(f"<{tag}>")[-1].split(f"</{tag}>")[0]

def get_submission_patches(evaluation, experiments_base_dir="resource/experiments/evaluation"):
    with open(f'{experiments_base_dir}/lite/{evaluation}/all_preds.jsonl', 'r') as f:
        res = {}
        for line in f:
            data = json.loads(line)
            res[data['instance_id']] = data
        return res

def get_resolved_set(result_path):
    with open(result_path, 'r') as f:
        data = json.load(f)
        if 'resolved' not in data:
            return set(data['resolved_ids'])
        return set(data['resolved'])

def get_submission_resolved_set(evaluation):
    try:
        with open(f'resource/experiments/evaluation/lite/{evaluation}/results/moatless_eval.json', 'r') as f:
            data = json.load(f)
            if 'resolved' not in data:
                return set(data['resolved_ids'])
            return set(data['resolved'])
    except FileNotFoundError:
        with open(f'resource/experiments/evaluation/lite/{evaluation}/results/results.json', 'r') as f:
            data = json.load(f)
            if 'resolved' not in data:
                return set(data['resolved_ids'])
            return set(data['resolved'])


evals = [
    #'20240612_MASAI_gpt4o',
    #'20240615_appmap-navie_gpt4o', # 21.67
    #'20240509_amazon-q-developer-agent-20240430-dev',# 20.33
    #'20240612_marscode-agent-dev',
   '20240523_aider',
    #'20240604_CodeR',
    #'20240524_opencsg_starship_gpt4',# 23.67
    #'20240612_IBM_Research_Agent101',
    #'20240617_moatless_gpt4o', # 24.67
    #'20240617_factory_code_droid',
    ### THESE ABOVE ARE THE ONES WE ORIGINALLY TESTED
    ### THE FOLLOWING ARE THE NEW ONES
    #'20240621_autocoderover-v20240620',
    #'20240622_Lingma_Agent',
     '20240623_moatless_claude35sonnet',
    #'20240627_abanteai_mentatbot_gpt4o',
     '20240630_agentless_gpt4o',
    #'20240702_codestory_aide_mixed',
    #'20240706_sima_gpt4o',
    '20240725_opendevin_codeact_v1.8_claude35sonnet'
]

# evals = [
#     #'20240702_codestory_aide_mixed', # 43
#     #'20240627_abanteai_mentatbot_gpt4o', # 38
#     #'20240622_Lingma_Agent', # 33
#     #'20240621_autocoderover-v20240620', # 30.67
#     #'20240604_CodeR', # 28.33,
#     #'20240612_MASAI_gpt4o', # 28.0
#     #'20240706_sima_gpt4o', # 27.67,
#     #'20240630_agentless_gpt4o', # 27.33
#     #'20230726_cosine_genie', # 50.67
# ]

# resolved_instances = {eval: get_resolved_set(eval) for eval in evals}
# patches = {eval: get_patches(eval) for eval in evals}

import datasets

def get_full_data():
    dataset = datasets.load_dataset('princeton-nlp/SWE-bench_Lite', split="test")
    return list(dataset)

def get_candidate_instances(dataset): # only rerank instances that have at least one correct patch
    global evals
    global resolved_instances
    dataset = datasets.load_dataset('princeton-nlp/SWE-bench_Lite', split="test")
    
    
    # all_resolved_instances = set()
    # for eval in evals:
    #     all_resolved_instances.update(resolved_instances[eval])
    
    return [instance for instance in dataset]# if instance['instance_id'] in all_resolved_instances]


def evaluate_ranking(instance_id, sorted_evals, resolved_instances, topk=[1,3,5]):
    res = {}
    for k in topk:
        is_pass = False
        for eval in sorted_evals[:k]:
            if instance_id in resolved_instances[eval]:
                is_pass = True
                break
        res[f'top {k} pass'] = is_pass
    return res


def patch_to_code_before_after(patch):
    lines = patch.split('\n')
    before = []
    after = []
    hunk_before = []
    hunk_after = []
    in_hunk = False

    prev_filename = None
    cur_filename = None
    file_names = []
    
    for line in lines:
        if line.startswith('--- a/'):
            prev_filename = cur_filename
            cur_filename = line[6:]
            continue
        if line.startswith('+++ b/'):
            continue
        if line.startswith('index'):
            continue
        if line.startswith('new file'):
            continue
        if line.startswith('deleted file'):
            continue
        if line.startswith('diff'):
            continue
        
        if line.startswith('@@'):
            if in_hunk:
                before.append('\n'.join(hunk_before))
                after.append('\n'.join(hunk_after))
                file_names.append(prev_filename)
                hunk_before = []
                hunk_after = []
            in_hunk = True
            line = line.split('@@')[2]
        if in_hunk:
            if line.startswith('-'):
                hunk_before.append(line[1:])
            elif line.startswith('+'):
                hunk_after.append(line[1:])
            else:
                hunk_before.append(line[1:])
                hunk_after.append(line[1:])

    if in_hunk:
        before.append('\n'.join(hunk_before))
        after.append('\n'.join(hunk_after))
        file_names.append(cur_filename)
    
    last_filename = None
    
    before_code = ""
    after_code = ""
    
    
    
    for i in range(len(file_names)):
        if file_names[i] != last_filename:
            if last_filename:
                before_code += f'\n</file="{last_filename}">\n'
            before_code += f'\n<file="{file_names[i]}">\n'
            before_code += before[i] + '\n ... \n ... \n'
            if last_filename:
                after_code += f'\n</file="{last_filename}">\n'
            after_code += f'\n<file="{file_names[i]}">\n'
            after_code += after[i] + '\n ... \n ... \n'
            last_filename = file_names[i]
        else:
            before_code += before[i] + '\n ... \n ... \n'
            after_code += after[i] + '\n ... \n ... \n'
            
    before_code += f'\n</file="{last_filename}">\n'
    after_code += f'\n</file="{last_filename}">\n'

    return before_code, after_code


import re

def patch_to_linenumbers(patch):
    lines = patch.split('\n')
    
    cur_filename_a = None
    cur_filename_b = None
    
    hunk_header_re = re.compile(r'@@ -(\d+),(\d+) \+(\d+),(\d+) @@')
    
    filename_re_a = re.compile(r'--- a/([\w/.\-_]+)')
    filename_re_b = re.compile(r'\+\+\+ b/([\w/.\-_]+)')
    
    res = []
    
    for line in lines:    
        if line.startswith('--- a/'):
            filename_match = filename_re_a.match(line)
            cur_filename_a = filename_match.group(1)
        elif line.startswith('+++ b/'):
            filename_match = filename_re_b.match(line)
            cur_filename_b = filename_match.group(1)
        
        hunk_header_match = hunk_header_re.match(line)
        
        if hunk_header_match:
            before_line_start = int(hunk_header_match.group(1))
            before_line_count = int(hunk_header_match.group(2))
            if before_line_count == 0 and before_line_start == 0:
                cur_filename_a = None # file does not exist before patching
            after_line_start = int(hunk_header_match.group(3))
            after_line_count = int(hunk_header_match.group(4))

            res.append({
                'filename_a': cur_filename_a,
                'filename_b': cur_filename_b,
                'before_line_start': before_line_start,
                'before_line_count': before_line_count,
                'after_line_start': after_line_start,
                'after_line_count': after_line_count,
            })
    
    return res


from repo import setup_swebench_repo, apply_patch

def get_before_after_code_with_context(instance: dict, patch, repo_base_dir: str, context_window=20):
    # get before and after patch hunks with an extended context window
    
    line_numbers = patch_to_linenumbers(patch)
    
    new_line_numbers = []
    for line_number in line_numbers:
        if (not line_number['filename_a'] or '.' + line_number['filename_a'].split('.')[-1] in code_file_extensions) \
            and (not line_number['filename_b'] or '.' + line_number['filename_b'].split('.')[-1] in code_file_extensions):
            new_line_numbers.append(line_number)
    
    line_numbers = new_line_numbers
    
    # extend the intervals by context_window on both sides
    
    for line_number in line_numbers:
        before_line_start = max(1, line_number['before_line_start'] - context_window)
        before_line_count = line_number['before_line_count'] + context_window * 2
        after_line_start = max(1, line_number['after_line_start'] - context_window)
        after_line_count = line_number['after_line_count'] + context_window * 2
        
        line_number['before_line_start'] = before_line_start
        line_number['before_line_count'] = before_line_count
        line_number['after_line_start'] = after_line_start
        line_number['after_line_count'] = after_line_count
        

    before_merged_linumbers = []
    
    for line_number in line_numbers:
        if not before_merged_linumbers:
            before_merged_linumbers.append(line_number)
        else:
            last_line_number = before_merged_linumbers[-1]
            if last_line_number["filename_a"] == line_number["filename_a"] and last_line_number["before_line_start"] + last_line_number["before_line_count"] >= line_number["before_line_start"]:
                last_line_number["before_line_count"] = max(last_line_number["before_line_start"] + last_line_number["before_line_count"], line_number["before_line_start"] + line_number["before_line_count"]) - last_line_number["before_line_start"]
            else:
                before_merged_linumbers.append(line_number)
    
    after_merged_linumbers = []
    
    for line_number in line_numbers:
        if not after_merged_linumbers:
            after_merged_linumbers.append(line_number)
        else:
            last_line_number = after_merged_linumbers[-1]
            if last_line_number["filename_b"] == line_number["filename_b"] and last_line_number["after_line_start"] + last_line_number["after_line_count"] >= line_number["after_line_start"]:
                last_line_number["after_line_count"] = max(last_line_number["after_line_start"] + last_line_number["after_line_count"], line_number["after_line_start"] + line_number["after_line_count"]) - last_line_number["after_line_start"]
            else:
                after_merged_linumbers.append(line_number)
    
    
    
    repo_dir = setup_swebench_repo(instance, repo_base_dir)
    
    before_code = ""
    
    for line_number in before_merged_linumbers:
        filename = line_number['filename_a']
        before_line_start = line_number['before_line_start']
        before_line_count = line_number['before_line_count']
        
        before_code += f'<before_file="a/{filename}" start_line={before_line_start}>\n'
        
        if not filename:
            before_code += "# THIS FILE DOES NOT EXIST BEFORE PATCHING"

        else:
            try:
                with open(f'{repo_dir}/{filename}', 'r') as f:
                    lines = f.readlines()
                    before_lines = lines[before_line_start-1:before_line_start+before_line_count-1]
                    numbered_lines = [f'({i+before_line_start:04d}) {line}' for i, line in enumerate(before_lines)]
                    before_code += ''.join(numbered_lines)
            except FileNotFoundError:
                import pdb; pdb.set_trace()
        
        before_code += f'</before_file="a/{filename}" start_line={before_line_start}>\n'
    
    after_code = ""
    
    apply_result = apply_patch(patch, repo_dir, instance) # apply the patch
    
    for line_number in after_merged_linumbers:
        filename = line_number['filename_b']
        after_line_start = line_number['after_line_start']
        after_line_count = line_number['after_line_count']
        
        after_code += f'<after_file="b/{filename}" start_line={after_line_start}>\n'
        
        if not filename:
            after_code += "# THIS FILE DOES NOT EXIST AFTER PATCHING"
        else:
            
            with open(f'{repo_dir}/{filename}', 'r') as f:
                lines = f.readlines()
                after_lines = lines[after_line_start-1:after_line_start+after_line_count-1]
                numbered_lines = [f'{i+after_line_start}: {line}' for i, line in enumerate(after_lines)]
                after_code += ''.join(numbered_lines)
        
        after_code += f'</after_file="b/{filename} start_line={after_line_start}">\n\n'

    apply_patch(patch, repo_dir, instance, revert=True)
    return before_code, after_code


def get_identified_spans(instance, trajectory_path):
    instance_id = instance["instance_id"]
    trajectory_path = os.path.join(trajectory_path, f"{instance_id}.json")
    
    identified_spans_set = set()
    
    with open(trajectory_path, 'r') as f:
        trajectory = json.load(f)
        for transition in trajectory["transitions"]:
            for action in transition["actions"]:
                if "identified_spans" in action['action']:
                    identified_spans = action['action']["identified_spans"]
                    
                    for span in identified_spans:
                        for span_id in span["span_ids"]:
                            identified_spans_set.add((span["file_path"], span_id))
                            
    
    identified_spans_set = sorted(list(identified_spans_set), key=lambda x: x[0])
    identified_spans = []
    
    for file_path, span_id in identified_spans_set:
        if not identified_spans or identified_spans[-1]["file_path"] != file_path:
            identified_spans.append({
                "file_path": file_path,
                "span_ids": [span_id]
            })
        else:
            identified_spans[-1]["span_ids"].append(span_id)
    
    return identified_spans
