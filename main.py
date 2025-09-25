#By Daniel Motilla (M0TH)

from lark import Lark, Transformer
from build123d import *
from ocp_vscode import show, show_all

grammar = """
start: add_expr
     | sub_expr
add_expr: NUMBER "+" NUMBER -> add_expr
sub_expr: NUMBER "-" NUMBER -> sub_expr
%import common.NUMBER
%ignore " "
"""

# Global variable to store the result
result_part = None

class CalcTransformer(Transformer):
    def add_expr(self, args):
        global result_part
        
        x = int(args[0])
        y = int(args[1])
        spacing = 3.0  # More space needed for varying sizes
        
        # Create boxes side by side with increasing sizes
        with BuildPart() as calc_result:
            # Create x boxes starting at position 0 with increasing sizes
            for i in range(x):
                size = 1.0 + i * 0.3  # Start at 1.0, increase by 0.3 each time
                with Locations((i * spacing, 0, 0)):
                    Box(size, size, size)
            
            # Create y boxes continuing the size increase
            for i in range(y):
                size = 1.0 + (x + i) * 0.3  # Continue size progression
                with Locations((x * spacing + i * spacing, 0, 0)):
                    Box(size, size, size)
        
        # Store the result globally so show_all() can find it
        result_part = calc_result.part
        result_part.label = f"addition_{x}_plus_{y}"
        
        return f"Created {x + y} boxes: {x} + {y} with increasing sizes"
    
    def sub_expr(self, args):
        global result_part
        
        x = int(args[0])
        y = int(args[1])
        spacing = 3.0  # More space needed for varying sizes
        
        # Create boxes with decreasing sizes
        with BuildPart() as calc_result:
            # Create x boxes starting at position 0 with decreasing sizes
            total_boxes = x + y
            for i in range(x):
                size = 2.0 - i * 0.2  # Start larger, decrease by 0.2 each time
                size = max(size, 0.3)  # Don't let it get too small
                with Locations((i * spacing, 0, 0)):
                    Box(size, size, size)
            
            # Create y boxes to the left continuing the size decrease
            for i in range(y):
                size = 2.0 - (x + i) * 0.2  # Continue size progression
                size = max(size, 0.3)  # Don't let it get too small
                with Locations((-spacing - i * spacing, 0, 0)):
                    Box(size, size, size)
        
        # Store the result globally so show_all() can find it
        result_part = calc_result.part
        result_part.label = f"subtraction_{x}_minus_{y}"
        
        return f"Created {x + y} boxes: {x} with {y} to the left, decreasing sizes"

parser = Lark(grammar, parser='lalr', transformer=CalcTransformer())

def main():
    # Parse and create geometry
    result = parser.parse("1 - 10")
    print(result)
    
    # Show the result
    if result_part is not None:
        print(f"Result part created: {result_part}")
        print(f"Part volume: {result_part.volume}")
        print(f"Part bounding box: {result_part.bounding_box()}")
        
        # Try different display methods
        try:
            show_all()
        except NameError:
            print("show_all not available")
        
        try:
            show(result_part)
        except NameError:
            print("show not available")
    else:
        print("No result_part created!")
    
    # Just to verify - let's also assign to a local variable
    boxes = result_part
    print(f"Local variable 'boxes' assigned: {boxes}")

if __name__ == '__main__':
    main()