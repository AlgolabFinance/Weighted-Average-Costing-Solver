import base64
import os
import zipfile

from convert_to_iif_v2 import *
import streamlit as st

def get_zip_download_link(path, file_name, description=''):
    with open(path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()
        file_name += '.zip'
        return f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download={file_name}>Download {description} ZIP</a>'



def zip_ya(startdir, file_name):
    z = zipfile.ZipFile(file_name,'w', zipfile.ZIP_DEFLATED)
    for dirpath, dirnames, filenames in os.walk(startdir):
        fpath = dirpath.replace(startdir,'')
        fpath = fpath and fpath + os.sep or ''
        for filename in filenames:
            z.write(os.path.join(dirpath, filename),fpath+filename)
    z.close()


if __name__ == '__main__':
    cgl_path = st.text_input('8949 Files Directory')
    mapping_file_path = st.text_input('Please inpout the path of the mapping file')
    pd.options.mode.chained_assignment = None
    if st.button('Process 8949 Data'):
        dir = cgl_path.strip()
        if len(dir) == 0:
            st.write('Please input directory')
        else:
            convert_all_files(dir + '*', True, mapping_file_path)
            output_path = os.getcwd()
            output_path += '/cgl_iif_output'
            zip_ya(dir, 'cgl_iif_output')
            with st.sidebar:
                st.markdown(get_zip_download_link(output_path,'cgl_iif_output', 'CGL IIF'), unsafe_allow_html=True)
    qb_path = st.text_input('Quickbook Files Directory')
    if st.button('Process Post Quickbook Data'):
        dir = qb_path.strip()
        if len(dir) == 0:
            st.write('Please input directory of quickbook files')
        else:
            convert_all_files(dir, False)
            output_path = os.getcwd()
            output_path += '/quickbook_iif_output'
            zip_ya(dir, 'quickbook_iif_output')
            with st.sidebar:
                st.markdown(get_zip_download_link(output_path, 'quickbook_iif_output', 'Quickbook IIF'), unsafe_allow_html=True)


# /Users/tonghaoyang/PycharmProjects/Weighted-Average-Costing-Solver2/quickbook/8949/
# /Users/tonghaoyang/PycharmProjects/Weighted-Average-Costing-Solver2/quickbook/QB/
# /Users/tonghaoyang/PycharmProjects/Weighted-Average-Costing-Solver2/quickbook/mapping.xlsx
