import numpy as np
from scipy.sparse import csr_matrix


def data_masks(all_usr_pois, item_tail):
    # make session sequences the same length using the tail '0'
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens)
    indptr, indices, data = [], [], []
    indptr.append(0)
    for j in range(len(all_usr_pois)):
        input = all_usr_pois[j]
        length = len(input)
        s = indptr[-1]
        indptr.append((s + length))
        for i in range(length):
            indices.append(i)
            data.append(input[i])
    matrix = csr_matrix((data, indices, indptr), shape=(len(all_usr_pois), len_max))
    indptr, indices, data = [], [], []
    indptr.append(0)
    return len_max, matrix


class Data():
    def __init__(self, data, n_node=None):
        inputs = data[0]
        len_max, matrix = data_masks(inputs, [0])
        self.len_max = len_max
        self.inputs = matrix
        self.inputs_length = matrix.shape[0]
        self.n_node = n_node

        self.len_max = len_max
        self.targets = np.asarray(data[1])
        self.length = len(inputs)


    def get_overlap(self, items):
        # get the incidence matrix of the line graph
        matrix = np.zeros((len(items), len(items)))
        for i in range(len(items)):
            seq_a = set(items[i])
            seq_a.discard(0)
            for j in range(i+1, len(items)):
                seq_b = set(items[j])
                seq_b.discard(0)
                overlap = seq_a.intersection(seq_b)
                ab_set = seq_a | seq_b
                matrix[i][j] = float(len(overlap))/float(len(ab_set))
                matrix[j][i] = matrix[i][j]

        matrix = matrix + np.diag([1.0]*len(items))
        degree = np.sum(np.array(matrix), 1)
        degree = np.diag(1.0/degree)
        return matrix, degree

    def generate_batch(self, batch_size):
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = np.arange(self.length-batch_size, self.length)
        return slices


    def get_slice(self, index):

        items, num_node = [], []
        last = []  # the last item id in each session
        item_set = []  # all the item set in this batch
        cnt = 0
        inp = self.inputs[index].toarray()
        for u_input in inp:
            num_node.append(len(np.nonzero(u_input)[0]))
            a = set(np.unique(u_input))
            a.discard(0)
            item_set += list(a)
        item_set = set(item_set)
        nn = len(item_set)  # the item number in this batch
        a = range(0, nn)
        id_dict = dict(zip(list(item_set), a))  # give the item a new id
        max_n_node = np.max(num_node)
        session_len = []
        data = []
        indices = tuple()
        inptr = [0]

        for u_input in inp:
            # construct the sparse hypergraph incidence matrix
            input = np.nonzero(u_input)[0]
            session_len.append([len(input)])
            node = set(np.unique(u_input))
            node.discard(0)
            keys = list(node)
            if len(keys) == 1:
                indices += tuple([id_dict[keys[0]]])
            else:
                from operator import itemgetter
                ind = sorted(itemgetter(*keys)(id_dict))
                indices += tuple(ind)
            data += [1] * len(keys)
            s = inptr[-1]
            inptr.append((s + len(keys)))
            cnt += 1
            # re-construct the session item by using the new item id
            node = u_input[np.nonzero(u_input)]
            last.append(id_dict[node[-1]])
            from operator import itemgetter
            node = itemgetter(*node)(id_dict)
            if len(np.nonzero(u_input)[0]) == 1:
                nod = [node]
            else:
                nod = list(node)
            node = [f+1 for f in nod]
            items.append(node + (max_n_node - len(node)) * [0])
        matrix = csr_matrix((data, list(indices), inptr), shape=(cnt, len(item_set)))
        H_T = matrix.toarray()
        H = np.transpose(H_T)
        D = np.diag(1.0/np.sum(H, axis=1))
        B = np.diag(1.0/np.sum(H, axis=0))
        id_dict1 = dict(zip(a, list(item_set)))
        sorted(id_dict1.keys())
        item_map = list(id_dict1.values())  # sort the item set by the new item id

        return items, self.targets[index]-1, last, \
               H, H_T, D, B, item_map, session_len,self.inputs[index].toarray()
