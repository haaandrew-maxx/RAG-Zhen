from dotenv import load_dotenv
load_dotenv()

from graph.graph import app
import pandas as pd

if __name__ == "__main__":
    print("This is the main module.")

    # Load description
    with open("description.txt", "r", encoding="utf-8") as f:
        description = f.read()

    # Load BOM Excel
    df = pd.read_excel("BOM.xlsx")
    bom = df.to_markdown(index=False)

    # Example question (you may modify this)
    question = "Ciclos materiales: acero, motores, contrapesos, electrónica, plásticos."

    result = app.invoke(
        input={
            "question": question,
            "bom": bom,
            "description": description,
        }
    )

    print(result)
