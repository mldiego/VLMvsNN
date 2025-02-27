# standard lib
import base64
import operator
import os
import random
import sys
import time

# third-party
from PIL import Image
import google.generativeai as genai

## Configure model to run
api_key = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=api_key)


DIGIT_CHOICES = [str(i) for i in range(10)]
OPERATOR_CHOICES = ['+', '-', '%', '*']
OPS = {'+': operator.add, '-': operator.sub, '*': operator.mul, '%': operator.truediv}


def vlm_sequence(model_str, sample_fp):
    """
    Given an arithmetic expression (as a list of images describing it), return the output of the arithmetic expression
    described by the operands and operators.

    model_str (str): the name of the VLM to use
    sample_fps (str): filepath of the directory containing the images (pngs) to give as input
    """
    image_fps = [os.path.join(sample_fp, image_fp) for image_fp in os.listdir(sample_fp)]

    input_prompt = '''
        You will be provided a sequence of input images. Contained in each image will be
        either a digit (0-9) or an operator (+, -, /, *). Read these together to make an
        arithmetic expression, identify the digit or operand in each image to write an 
        arithmetic expression. Then, solve the arithmetic expression from left to right. For
        example, given a sequence of images like ['image1': '5', 'image2:'+', 'image3':'1', 
        'image4':'*', 'image5':'2], you would first read the valid expression '5+1' and 
        evaluate it to 6. Then, you would read the next operator and operand to get the 
        valid expression '6*2', which is evaluated to 12. 

        The output must be in the format <numerical expression in image>=<solution>. 
        DO NOT give me any other output. Also, DO NOT use LaTeX. These are simple expressions 
        and can be expressed without the use of LaTeX.

        From the example above, the only output needed is: "5+1*2=12"
        
    '''

    ## Select model
    model = genai.GenerativeModel(model_name = model_str)

    contents = [input_prompt] + image_fps
    # print(contents)

    start = time.time()
    res = model.generate_content(contents=contents)
    end = time.time()

    return res.text, end - start


def get_vlm_output_long_expression(model_str, num_operands, num_samples):
    """
    Given a set of long arithmetic expressions (# operands > 2), evaluate them with the neural automaton
    and the VLM and compare the results.

    model_str (str): the name of the VLM model to use
    """
    vlm_sequence_fp = os.path.join('sequence', str(num_operands))
    results_fp = os.path.join('results/sequence', str(num_operands))
    labels_fp = os.path.join('labels', f'{num_operands}_labels.txt')

    samples = []

    with open(labels_fp, 'r') as f:
        for sample in os.listdir(vlm_sequence_fp):
            sample_fp = os.path.join(vlm_sequence_fp, sample) # /path/to/vlm/sequence/{num_operands}/sample_0
            line = f.readline()
            samples.append([sample_fp] + [sample.strip() for sample in line.split(',')])

    results_file = os.path.join(results_fp, f'results_{model_str}.txt')

    for sample_num, (sample_fp, true_expression, true_solution) in enumerate(samples):
        if sample_num > num_samples: # only first 50 samples
            continue
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        with open(results_file, 'a') as f:
            res, time_taken = vlm_sequence(model_str, sample_fp)
            print(f'#{sample_num} -> {true_expression}={true_solution} | {res}: {time_taken}')
            f.write(f'#{sample_num} -> {true_expression}={true_solution} | {res}: {time_taken}\n')
            time.sleep(5)  # Pause execution for 5 second to not exceed gemini quota (15 request per minute, 1500 requests per day)


if __name__=="__main__":

    models = [
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash",
        "gemini-1.5-pro"
    ]

    num_samples = 25

    for model in models:
        # for num_operands in range(3, 11):
        for num_operands in range(3, 11):
            get_vlm_output_long_expression(model, num_operands, num_samples)