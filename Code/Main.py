
import yaml
from SubjectProcessor import SubjectProcessor

# Load configuration file
cfg = yaml.safe_load(open("config.yaml", "r"))

# Process each subject
for subj in cfg["subjects"]:
    processor = SubjectProcessor(subj, cfg)
    processor.process_all()