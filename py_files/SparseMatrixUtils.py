from scipy import sparse

class SparseMatrixUtils:
    '''
    ''' 
    
    @staticmethod
    def save_sparse_matrix(sparse_matrix, sparse_matrix_name, matrix_dir = '../data/sparse_matrix'):
        ''' Saves the sparse matrix to the matrix directory and returns the file path
        '''
        sparse_matrix_name += '.npz'
        f = matrix_dir + '/' + sparse_matrix_name
        sparse.save_npz(f, sparse_matrix)

        return f 
    
    @staticmethod
    def load_sparse_matrix(sparse_matrix_name, matrix_dir = '../data/sparse_matrix'):
        ''' Loads a sparse matrix from the matrix directory
        '''
        f = matrix_dir + '/' + sparse_matrix_name + '.npz'
        return sparse.load_npz(f)