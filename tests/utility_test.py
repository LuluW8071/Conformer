import sys
import time  

from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils import TextTransform

text_transform = TextTransform()


def transformation(text):
    # Measure time taken for text to int conversion
    start_time = time.time()
    int_seq = text_transform.text_to_int(text.lower())
    end_time = time.time()

    # Print the integer sequence and the time taken
    print("Integer sequence:", int_seq)
    print("Time taken to convert text to integers: {:.6f} seconds".format(end_time - start_time))


    # Measure time taken for int to text conversion
    start_time = time.time() 
    back_to_text = text_transform.int_to_text(int_seq)
    end_time = time.time()

    # Print the reconstructed text and the time taken
    print("\nReconstructed text:", back_to_text)
    print("Time taken to convert integers back to text: {:.6f} seconds".format(end_time - start_time))


if __name__ == "__main__":
    # Example sentence
    text = "Màny people in white shirts are walking down à street. So, I might wear white shirt as well."
    transformation(text)