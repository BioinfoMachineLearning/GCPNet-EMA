import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.app import app

if __name__ == "__main__":
    app.run()
