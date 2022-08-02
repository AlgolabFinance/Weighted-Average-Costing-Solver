from convert_to_iif import *
import streamlit as st

if __name__ == '__main__':
    cgl_path = st.text_input('Post 8949 Files Directory')
    if st.button('Process Post 8949 Data'):
        dir = cgl_path.strip()
        if len(dir) == 0:
            st.write('Please input directory')
        else:
            convert_all_files(dir, True)
    qb_path = st.text_input('Quickbook Files Directory')
    if st.button('Process Post Quickbook Data'):
        dir = qb_path.strip()
        if len(dir) == 0:
            st.write('Please input directory')
        else:
            convert_all_files(dir, False)

# # /Users/tonghaoyang/PycharmProjects/Weighted-Average-Costing-Solver2/quickbook/8949/
# # /Users/tonghaoyang/PycharmProjects/Weighted-Average-Costing-Solver2/quickbook/QB/
