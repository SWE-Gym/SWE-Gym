# from DEI paper
# modified from https://github.com/aorwall/moatless-tools/blob/10e2c61e187374a0195d9167fa0658497bdc4a98/moatless/utils/repo.py

import logging
import os
import subprocess
import re

logger = logging.getLogger(__name__)

def setup_github_repo(repo: str, base_commit: str, base_dir: str = "./tmp/repos") -> str:
    repo_name = get_repo_dir_name(repo)
    repo_url = f"https://github.com/{repo}.git"

    path = f"{base_dir}/{repo_name}"
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(f"Directory '{path}' was created.")
    maybe_clone(repo_url, path)
    checkout_commit(path, base_commit)

    return path


def get_repo_dir_name(repo: str):
    return repo.replace("/", "_")


def maybe_clone(repo_url, repo_dir):
    if not os.path.exists(f"{repo_dir}/.git"):
        logger.info(f"Cloning repo '{repo_url}'")
        # Clone the repo if the directory doesn't exist
        result = subprocess.run(
            ["git", "clone", repo_url, repo_dir],
            check=True,
            text=True,
            capture_output=True,
        )

        if result.returncode == 0:
            logger.info(f"Repo '{repo_url}' was cloned to '{repo_dir}'")
        else:
            logger.info(f"Failed to clone repo '{repo_url}' to '{repo_dir}'")
            raise ValueError(f"Failed to clone repo '{repo_url}' to '{repo_dir}'")


def pull_latest(repo_dir):
    subprocess.run(
        ["git", "pull"],
        cwd=repo_dir,
        check=True,
        text=True,
        capture_output=True,
    )


def clean_and_reset_state(repo_dir):
    subprocess.run(
        ["git", "clean", "-fd"],
        cwd=repo_dir,
        check=True,
        text=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "reset", "--hard"],
        cwd=repo_dir,
        check=True,
        text=True,
        capture_output=True,
    )


def create_branch(repo_dir, branch_name):

    try:
        subprocess.run(
            ["git", "branch", branch_name],
            cwd=repo_dir,
            check=True,
            text=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        print(e.stderr)
        raise e


def create_and_checkout_branch(repo_dir, branch_name):
    try:
        branches = subprocess.run(
            ["git", "branch"],
            cwd=repo_dir,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.split("\n")
        branches = [branch.strip() for branch in branches]
        if branch_name in branches:
            subprocess.run(
                ["git", "checkout", branch_name],
                cwd=repo_dir,
                check=True,
                text=True,
                capture_output=True,
            )
        else:
            output = subprocess.run(
                ["git", "checkout", "-b", branch_name],
                cwd=repo_dir,
                check=True,
                text=True,
                capture_output=True,
            )
    except subprocess.CalledProcessError as e:
        print(e.stderr)
        raise e


def commit_changes(repo_dir, commit_message):
    subprocess.run(
        ["git", "commit", "-m", commit_message],
        cwd=repo_dir,
        check=True,
        text=True,
        capture_output=True,
    )


def checkout_branch(repo_dir, branch_name):
    subprocess.run(
        ["git", "checkout", branch_name],
        cwd=repo_dir,
        check=True,
        text=True,
        capture_output=True,
    )


def push_branch(repo_dir, branch_name):
    subprocess.run(
        ["git", "push", "origin", branch_name],
        cwd=repo_dir,
        check=True,
        text=True,
        capture_output=True,
    )


def get_diff(repo_dir):
    output = subprocess.run(
        ["git", "diff"], cwd=repo_dir, check=True, text=True, capture_output=True
    )

    return output.stdout


def stage_all_files(repo_dir):
    subprocess.run(
        ["git", "add", "."], cwd=repo_dir, check=True, text=True, capture_output=True
    )


def checkout_commit(repo_dir, commit_hash):
    subprocess.run(
        ["git", "reset", "--hard", commit_hash],
        cwd=repo_dir,
        check=True,
        text=True,
        capture_output=True,
    )


def setup_repo(repo_url, repo_dir, branch_name="master"):
    maybe_clone(repo_url, repo_dir)
    clean_and_reset_state(repo_dir)
    checkout_branch(repo_dir, branch_name)
    pull_latest(repo_dir)


def clean_and_reset_repo(repo_dir, branch_name="master"):
    clean_and_reset_state(repo_dir)
    checkout_branch(repo_dir, branch_name)
    pull_latest(repo_dir)


def setup_swebench_repo(instance_data: dict, repo_base_dir: str = "./tmp/repos") -> str:
    repo_dir_name = instance_data["repo"].replace("/", "__")
    repo_base_dir += f"/{instance_data['instance_id']}"
    swebench_repos = {
        "astropy",
        "django",
        "matplotlib",
        "seaborn",
        "requests",
        "pylint",
        "pytest",
        "scikit-learn",
        "sphinx",
        "sympy"
    }
    github_repo_path = f"swe-train/{repo_dir_name}"
    for repo in swebench_repos:
        if repo in repo_dir_name:
            github_repo_path = f"swe-bench/{repo_dir_name}"
            break
    return setup_github_repo(
        repo=github_repo_path,
        base_commit=instance_data["base_commit"],
        base_dir=repo_base_dir,
    )
    
    

# modified from https://github.com/aorwall/SWE-bench-docker/blob/main/swebench_docker/context_manager.py#L177

class ExecWrapper:
    def __init__(
        self,
        subprocess_args: dict = None,
        logger: logging.Logger = None,
    ):
        self.logger = logger
        if subprocess_args is None:
            self.subprocess_args = {}
        else:
            self.subprocess_args = subprocess_args

    def __call__(self, cmd, raise_error=True, **kwargs):
        try:
            # if isinstance(cmd, list):
            #     self.logger.info(f"Command: {' '.join(cmd)}", level=DEBUG)
            # else:
            #     self.logger.write(f"Command: {cmd}", level=DEBUG)
            combined_args = {**self.subprocess_args, **kwargs}
            # self.logger.write(f"Subprocess args: {json.dumps(combined_args)}", level=DEBUG)
            output = subprocess.run(cmd, **combined_args)
            # self.logger.write(f"Std. Output:\n{output.stdout}", level=DEBUG)
            # if output.stderr:
            #     self.logger.write(f"Std. Error:\n{output.stderr}", level=DEBUG)
            # self.logger.write(f"Return Code: {output.returncode}", level=DEBUG)
            return output
        except subprocess.CalledProcessError as e:
            if raise_error and self.logger is not None:
                self.logger.error(f"Error: {e}")
                # self.logger.write(f"Error: {e}", level=ERROR)
                # self.logger.write(f"Error stdout: {e.stdout}", level=ERROR)
                # if e.stderr:
                #     self.logger.write(f"Error stderr: {e.stderr}", level=ERROR)
                # self.logger.write(f"Error traceback: {format_exc()}", level=ERROR)
                raise e

def apply_patch(
    patch: str, repo_dir, instance: dict, patch_type="test", revert: bool = False
) -> bool:
    """
    Apply patch to task environment

    Args:
        patch (str): Plaintext of patch to apply
        patch_type (str): Type of patch (e.g. "eval", "test")
    Returns:
        bool: True if patch applied successfully, False otherwise
    """
    instance_id = instance["instance_id"]
        
    init_diff_patch_path = os.path.join(
        os.path.dirname(repo_dir.rstrip("/")),
        f"temp_{instance_id}_{patch_type}_init.patch",
    )
    
    exec_wrapper = ExecWrapper(
            subprocess_args={
                "cwd": repo_dir,
                "check": True,
                "shell": False,
                # "capture_output": False,
                "universal_newlines": True,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT,
            },
            logger=logger,
        )
    
    exec_wrapper(f"git diff > {init_diff_patch_path}", shell=True)

    # If patch is `None`, indicate in log and skip
    if patch is None:
        logger.error(f"Patch is `None` ({patch_type})")
        logger.error(f"APPLY_PATCH_FAIL; Prediction patch is `None`")
        return False

    # Write patch to temporary patch file in parent directory
    patch_path = os.path.join(
        os.path.dirname(repo_dir.rstrip("/")),
        f"temp_{instance_id}_{patch_type}.patch",
    )

    with open(patch_path, "w") as f:
        f.write(patch)

    # # Restore test files before applying if patch_type is 'test'
    # if patch_type == "test":
    #     for test in instance["test_directives"]:
    #         if os.path.exists(test):
    #             exec_wrapper(f"git restore {test}".split(" "))

    # Apply patch to testbed directory
    
    exclude_file = []
    
    for line in patch.split("\n"):
        m = re.match(r'^Binary files .* and b/(.*) differ', line)
        if m:
            binary_file = m.group(1)
            exclude_file.append(binary_file)
            
    apply_exclude_code = " ".join([f"--exclude {file}" for file in exclude_file])
    
    if exclude_file:    
        apply_cmd = (
            f"git apply -v -R {apply_exclude_code} {patch_path}" if revert \
                else f"git apply -v {apply_exclude_code} {patch_path}"
        )
    else:
        apply_cmd = (
            f"git apply -v -R {patch_path}" if revert \
                else f"git apply -v {patch_path}"
        )
        
    out_patch = exec_wrapper(apply_cmd.split(" "), raise_error=False, check=False)

    # If git command fails, try patch command
    if out_patch.returncode != 0:
        # Patch may has been partially applied so we should revert it.
        # NOTE: we do not revert the test patch because it may unintentionally revert previously applied patches
        if patch_type != "test":
            exec_wrapper("git restore .".split(" "))
            # revert to the state of the repo before the patch was applied
            output = exec_wrapper(f"git apply {init_diff_patch_path}".split(), raise_error=False, check=False)
            logger.info(f"Output (git apply - revert to initial state): {output.stdout}")
        apply_cmd = (f"patch -R --batch --fuzz=5 -p1 -i {patch_path}" if revert
                        else f"patch --batch --fuzz=5 -p1 -i {patch_path}")
        out_patch = exec_wrapper(apply_cmd.split(" "), raise_error=False, check=False)

    # TODO os.remove(patch_path)

    log_cmd = "Revert" if revert else "Apply"
    if out_patch.returncode != 0:
        # Patch apply failed
        logger.error(f"{log_cmd} patch failed ({patch_type})")
        logger.info(f"APPLY_PATCH_FAIL; ({patch_type})\nOutput:\n")
        logger.info(out_patch.stdout)
        if out_patch.stderr:
            logger.info(out_patch.stderr)
        if patch_type != "test" and "patching" in out_patch.stdout:
            # Patch has been partially applied so we should revert it.
            exec_wrapper("git restore .".split(" "))
            # revert to the state of the repo before the patch was applied
            output = exec_wrapper(f"git apply {init_diff_patch_path}".split(), raise_error=False, check=False)
            logger.info(f"Output (git apply - revert to initial state): {output.stdout}")
        return False

    # Patch apply succeeded
    logger.info(f"{log_cmd} patch successful ({patch_type})")
    logger.info(f"APPLY_PATCH_PASS ({patch_type})\n")
    return True