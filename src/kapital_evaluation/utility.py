
def format_options(options, dstype):
    if dstype == 'tc':
        # For 'tc', format by replacing ", " with ",\n"
        return options.replace(", ", ",\n")
    
    elif dstype == 'kmc': # check it
        # For 'kmc', format options with letters and new lines
        return ",\n".join([f"{chr(65 + i)}) {option}" for i, option in enumerate(options)])
    
    elif dstype == 'qmc': # check it
        # For 'qmc', format options with letters and new lines
        return ",\n".join([f"{chr(65 + i)}) {option}" for i, option in enumerate(options)])
    
    elif dstype == 'arc':
        # For 'arc', format sentences with letters and ensure no trailing periods
        return ',\n'.join([f"{chr(65 + i)}) {sentence.rstrip('.')}" for i, sentence in enumerate(options)])
    
    elif dstype == 'mc':
        # For 'mc', format by replacing ", " with ",\n"
        return options.replace(", ", ",\n")
    
    else:
        raise ValueError(f"Unsupported dataset type: {dstype}")


