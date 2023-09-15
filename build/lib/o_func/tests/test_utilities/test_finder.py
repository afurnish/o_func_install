from o_func import opsys3, finder ; start_path = opsys3()

def test_finder():
    # Define the line of code to search for
    line_to_search = "sossheig"
    # Define the starting directory to search in
    starting_dir = start_path + "GitHub"
    
    finder(line_to_search, starting_dir)