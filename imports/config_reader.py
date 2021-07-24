# --------------------------------------------------------------------
# A module for reading the configuration files, providing safe 
# parsing to the Python types.
# 
# author: Adam Kurowski
# mail:   akkurowski@gmail.com
# date:   13.07.2021
# --------------------------------------------------------------------

import configparser as cfg
import ast

# Save evaluation of literals to Python types.
def save_eval(expression):
    try:
        return ast.literal_eval(expression)
    except:
        pass
    try:
        prs_tree = ast.parse(expression, mode="eval")
    except:
        return expression
    
    allowed_node_types = []
    allowed_node_types.append(ast.Expression)
    allowed_node_types.append(ast.Num)
    allowed_node_types.append(ast.BinOp)
    allowed_node_types.append(ast.Pow)
    allowed_node_types.append(ast.Div)
    allowed_node_types.append(ast.Mult)
    allowed_node_types.append(ast.Add)
    allowed_node_types.append(ast.List)
    allowed_node_types.append(ast.Load)
    allowed_node_types.append(ast.Mod)
    allowed_node_types.append(ast.Tuple)
    allowed_node_types.append(ast.NameConstant)
    allowed_node_types.append(ast.Name)
    allowed_node_types.append(ast.Dict)
    allowed_node_types.append(ast.Constant)
    
    all_operations_are_allowed = True
    for e in ast.walk(prs_tree):
        all_operations_are_allowed = all_operations_are_allowed and (type(e) in allowed_node_types)
        
    if all_operations_are_allowed:
        try:
            return eval(expression)
        except:
            return expression
    else:
        raise RuntimeError(f'the save_eval function found a potentially unsafe-to-evaluate expression: ({expression})')

# Single obtaining action of a given setting from a source dictionary.
def obtain_setting(config, section, name):
    raw_config_string = config[section][name]
    try:
        parsed_cf_str = save_eval(raw_config_string)
        return parsed_cf_str
    except RuntimeError as e:
        print()
        print(f"parsing failed for this string: {raw_config_string}, due to this error: {e}")
        return raw_config_string

# A procedure for reading the settings structure from the specified
# path.
def obtain_settings_structure(settings_fname):
    config = cfg.ConfigParser(inline_comment_prefixes="#")
    config.read(settings_fname, encoding='utf-8')
    
    output_settings_structure = {}
    
    for section_name in config.keys():
        output_settings_structure.update({section_name:{}})
        for setting_name in config[section_name].keys():
            output_settings_structure[section_name].update({setting_name:obtain_setting(config, section_name, setting_name)})
    
    return output_settings_structure

# A debugging procedure for readable printing the contents and inferred types
# of the obtained settings file content.
def print_settings(settings):
    for section_name in settings.keys():
        print(f"[{section_name}]")
        for setting_name in settings[section_name].keys():
            print(f"\t{setting_name}:")
            setting_values = settings[section_name][setting_name]
            print(f"\t\t{setting_values} (type: {type(setting_values)})")
            print()