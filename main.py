# main.py
import sys
import argparse
from PyQt6.QtWidgets import QApplication

'''
# 训练
python main.py --mode real_train

# 测试
python main.py --mode test
'''

def main():
    parser = argparse.ArgumentParser(description='Ball Reaching System')
    parser.add_argument('--mode', type=str, default='sim_train',
                       choices=['real_train', 'test'],
                       help='Running mode')
    
    args = parser.parse_args()
    
    app = QApplication(sys.argv)

    if args.mode == 'real_train':
        from visualization.real_training_gui import RealTrainingGUI
        gui = RealTrainingGUI()
    elif args.mode == 'test':
        from visualization.test_gui import TestGUI
        gui = TestGUI()
    
    gui.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()