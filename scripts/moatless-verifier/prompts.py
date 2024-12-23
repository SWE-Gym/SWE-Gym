
DEI_PROMPT_TEMPLATE = """\
I want you to evaluate an LLM-generated candidate patch that tries to resolve an issue in a codebase.

To assist you in this task, you are provided with the following information:
 - You are given an issue text on a github repository (wrapped with <issue_description></issue_description>).
 - You are also given some identified code spans that are relevant to the issue.
    Each code span is wrapped with <code_span file_path=FILE_PATH span_id=SPAN_ID></code_span> tags, where FILE_PATH is the path to the file containing the code span, and SPAN_ID is the unique identifier for the code span.
    Each code span also comes with the line numbers for you to better understand the context. It's possible that the code span are not sufficient to fix the issue, adjust your score accordingly.
 - You are given the candidate patch that tries to resolve the target issue.
    For your convenience, you are given the hunks of original code and the code after applying the patch.
    The code before the patch is wrapped with <before_patch></before_patch> and the code after the patch is wrapped with <after_patch></after_patch>.
    Note that the file names in before_patch starts with 'a/' and the file names in after_patch starts with 'b/'.

Here's what you want to do:

1. Understand the issue. Explain in your own words what the issue is about. Output your explanation in <issue_exp></issue_exp> tags.
2. Understand the identified code spans. First provide a list of the span ids. Then explain how each of the identified code spans are relevant to the issue. Output your explanation in <code_span_exp></code_span_exp> tags.
3. Understand the candidate patch. First curate a list of modified hunks. For each modified hunk, explain what it's doing. Output your explanation in the <patch_exp></patch_exp> field.
4. Check if the patch is fixing the correct function or not. Output your explanation in the <correct_location_exp></correct_location_exp> field.
5. Check if the patch is introducing any new issues, especially if it contradicts with any of the identified code spans. Output your explanation in the <new_issues_exp></new_issues_exp> field.
6. Check if the patch can fix the issue. Compare the generated patch agains the common mistakes made by LLMs and see if it falls into any of the categories. Be ruthless to point out any potential mistakes. Output your explanation in the <fix_issue_exp></fix_issue_exp> field.
7. Finally, give me your score. Wrap your score in <score></score> tags. Make sure to include in these tags only an integer, nothing else.

Here's the scoring rubric:

Your score should be an integer between 0 and 10, where higher scores indicate better quality.
You should give a score of -1 if you think the patch is invalid or there is something wrong with it.
For every contradiction between the identified code spans and the patch, you should deduct 1 point from the score.
If you think the patch is not fixing the correct function, you should give a 0.
If you think the patch is introducing new issues, you should deduct 2 points from the score.
Your scoring should only be about the correctness of the patch, not about its quality or style.

<issue_description>
{issue_text}
</issue_description>

<before_patch>
{before_patch}
</before_patch>

<after_patch>
{after_patch}
</after_patch>

{code_spans}

Again, make sure your output ends with <score></score> tags containing only an integer.
For example, if your score is 8, the final part of output should look like this:
<score>8</score>
It should not contain any other information or characters.
Do not use ``` or ### or anything else to wrap your score.
"""


SYSTEM_PROMPT = """\
You are an expert in python for software engineering and code review. Your responsibility is to review the patches generated by language models to fix some issues and provide feedback on the quality of their code.
"""

SIMPLE_PROMPT_TEMPLATE = """\
I want you to evaluate an LLM-generated candidate patch that tries to resolve an issue in a codebase.

To assist you in this task, you are provided with the following information:
 - You are given an issue text on a github repository (wrapped with <issue_description></issue_description>).
 - You are also given some identified code spans that are relevant to the issue.
    Each code span is wrapped with <code_span file_path=FILE_PATH span_id=SPAN_ID></code_span> tags, where FILE_PATH is the path to the file containing the code span, and SPAN_ID is the unique identifier for the code span.
    Each code span also comes with the line numbers for you to better understand the context. It's possible that the code span are not sufficient to fix the issue, adjust your score accordingly.
 - You are given the candidate patch that tries to resolve the target issue.
    For your convenience, you are given the hunks of original code and the code after applying the patch.
    The code before the patch is wrapped with <before_patch></before_patch> and the code after the patch is wrapped with <after_patch></after_patch>.
    Note that the file names in before_patch starts with 'a/' and the file names in after_patch starts with 'b/'.


<issue_description>
{issue_text}
</issue_description>

<before_patch>
{before_patch}
</before_patch>

<after_patch>
{after_patch}
</after_patch>

{code_spans}

Response in "True" or "False" for whether the patch has resolved the issue.
"""