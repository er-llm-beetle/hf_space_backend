import re


def format_options(options, dstype):
    if dstype == 'tc':
        # For 'tc', format by replacing ", " with ",\n"
        formatted_options = options.replace(", ", ",\n")
        print("OPTIONS:\n", formatted_options)

        return formatted_options
    

    elif 'kmc' in dstype: # check it
        cleaned_text = options.replace("' ", "',")

        # Convert the string to a Python list
        items = eval(cleaned_text)

        # Iterate over the list and format it as needed
        formatted_items = [f"{chr(65 + i)}) {item}" for i, item in enumerate(items)]

        # Join the formatted items into a string with commas
        options = ",\n".join(formatted_items)

        # Print formatted options
        print("OPTIONS:\n", options)

        return options
    
    elif dstype == 'mmc': 
        formatted_options = options.replace(", ", ",\n")
        print("OPTIONS:\n", formatted_options)

        return formatted_options

    elif dstype == 'arc':
        # For 'arc', format sentences with letters and ensure no trailing periods
        options = eval(options) # from text to list
        formatted_options = ',\n'.join([f"{chr(65 + i)}) {sentence.rstrip('.')}" for i, sentence in enumerate(options)])
        print("OPTIONS:\n", formatted_options)
            
        return formatted_options
    
    elif dstype == 'mc':
        # For 'mc', format by replacing ", " with ",\n"
        split_options = re.split(r'(?<=,)\s*(?=[A-D]\))', options)

        # Format each option
        formatted_options = []
        for opt in split_options:
            # Remove leading/trailing whitespace
            opt = opt.strip()
            # Ensure there's a comma at the end if not already present
            if not opt.endswith(','):
                opt += ','
            formatted_options.append(opt)

        # Join with newlines
        result = '\n'.join(formatted_options)
        print("OPTIONS:\n", result)

        return result

    elif dstype == 'qmc': # fix it
        return ''

    else:
        raise ValueError(f"Unsupported dataset type: {dstype}")


