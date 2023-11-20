import os
encoder_path = "../../../../../../srv/scratch/bic/peter/model_save/Encoder2-99-2.95.pt"
if os.path.exists(encoder_path):
    # load
    print(f"found")
    pass
else:
    print(f"no")