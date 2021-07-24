#------------------------------------------------------------------------------#
# Procedures for communication with the user through the text interface
#
# author: Adam Kurowski
# mail:   akkurowski@gmail.com
# date:   13.07.2020
#------------------------------------------------------------------------------#


# Function for communicating with the script user.
# Used to ask a question with a yes/no answer returned as
# a logical variable
def ask_for_user_preference(question):
    while True:
        user_input = input(question+' (y/n): ')
        if user_input == 'y':
            return True
            break
        if user_input == 'n':
            return False
            break

# A function used to communicate with the user, which asks to
# choose one of several available options (by entering its number).
def ask_user_for_an_option_choice(question, val_rec_prompt, items, single_choice=True, indentation_level=1):
    def make_indentation(text, indentation_level):
        if indentation_level<0: indentation_level=0
        indented_text = ''
        for i in range(indentation_level):
            indented_text = '\t' + indented_text
        indented_text = indented_text+text
        return indented_text
    
    print(make_indentation(question, indentation_level-1))
    allowed_numbers = []
    for it_num, item in enumerate(items):
        print(make_indentation('(%i)'%(it_num+1)+str(item), indentation_level))
        allowed_numbers.append(str(it_num+1))
    while True:
        user_input  = input(make_indentation(val_rec_prompt, indentation_level))
        # We allow either single responses or multiple responses
        if single_choice:
            if user_input in allowed_numbers:
                return items[int(user_input)-1]
        else:
            split_input = user_input.split(' ')
            if len(split_input) == 1:
                if user_input in allowed_numbers:
                    return [items[int(user_input)-1]]
            else:
                output_options = []
                for sub_input in split_input:
                    if sub_input in allowed_numbers:
                        output_options.append(items[int(sub_input)-1])
                if len(output_options) != 0:
                    return output_options

# A function to communicate with the user that asks for a
# by the user of a floating point value.
def ask_user_for_a_float(question):
    while True:
        user_input = input(question)
        try:
            float_value = float(user_input)
            return float_value
            break
        except:
            pass

# A function to communicate with the user that asks for a
# by the user a string of characters.
def ask_user_for_a_string(question):
    while True:
        user_input = input(question)
        try:
            string_val = str(user_input)
            return string_val
            break
        except:
            pass