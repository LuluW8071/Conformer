import time  

from conformer.utils.logging import logger
from conformer.utils.text_transform import TextTransform


text_transform = TextTransform()


def transformation(text):
    # Measure time taken for text to int conversion
    start_time = time.time()
    int_seq = text_transform.text_to_int(text.lower())
    end_time = time.time()

    # Print the integer sequence and the time taken
    logger.info("Integer sequence: {}", int_seq)
    logger.info("Time taken to convert text to integers: {:.6f} seconds".format(end_time - start_time))


    # Measure time taken for int to text conversion
    start_time = time.time() 
    back_to_text = text_transform.int_to_text(int_seq)
    end_time = time.time()

    # Print the reconstructed text and the time taken
    logger.info("Reconstructed text: {}", back_to_text)
    logger.info("Time taken to convert integers back to text: {:.6f} seconds".format(end_time - start_time))


if __name__ == "__main__":
    # Example sentence
    text = "Màny people in white shirts are walking down à street So I might wear white shirt as well."
    transformation(text)