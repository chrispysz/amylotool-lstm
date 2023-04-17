import os
import asmscanlstm 

def main():
    lstm = ASMscanLSTM()
    
    # Predict for given sequence 
    prob, frag = lstm.predict("MKGRAFGHGRTYQAGGDLTVHEAAVFAPVGQVAAPPGT")
    print(prob, frag)

if __name__ == "__main__":
    main()