import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import git
def lsremote(url):
    remote_refs = {}
    g = git.cmd.Git()
    for ref in g.ls_remote(url).split('\n'):
        hash_ref_list = ref.split('\t')
        remote_refs[hash_ref_list[1]] = hash_ref_list[0]
    return remote_refs

refs = lsremote('https://bitbucket.org/aurorax/datangi')



