# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for GCPNet-EMA (https://github.com/BioinfoMachineLearning/GCPNet-EMA):
# -------------------------------------------------------------------------------------------------------------------------------------

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.app import app

if __name__ == "__main__":
    app.run()
