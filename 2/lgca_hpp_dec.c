/**
 * Клеточный автомат симуляции газа (Lattice gas automata (LGCA)).
 * Алгоритм взаимодействия HPP, прямоугольная решетка.
 * 
 * https://en.wikipedia.org/wiki/Lattice_gas_automaton
 * https://en.wikipedia.org/wiki/HPP_model
 * 
 * Для хранения состояния используется восьмибитное число.
 * Первые четыре бита используются для хранения частиц.
 * 
 * Номера битов:
 *         *
 *         |
 *         0
 *         |
 * *---3---*---1---*
 *         |
 *         2
 *         |
 *         *
 * 
 * @author Nikolay Khokhlov <khokhlov.ni@mipt.ru>, 2021
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <string.h>

#define ind(i, j) (((i + l->xmax) % l->xmax) + ((j + l->ymax) % l->ymax) * (l->xmax))

/*
 * Специальный маркер гля граничного условия стенки.
 */
#define WALL 16

typedef struct {
    int xmax;
    int ymax;
    unsigned char *lattice;
    unsigned char *lattice_buf;
    
    /* Декомпозиция. */
    //int start;
    //int finish;
    int rank; /* Номер процесса, k */
    int num_tasks; /* Число процессов, P*/
    MPI_Datatype block_type;
    MPI_Datatype column_type;

    /* Декартовая топология */
    int start[2];
    int finish[2]; 
    int coords[2]; /* Координаты в декартовой топологии */
    int dims[2]; /* Размер декартовой топологии */
    MPI_Comm cart_comm; /* Коммуникатор, описывающий декартову топологию */
    MPI_Comm row_comm; /* Коммуникатор строчки. */
    MPI_Comm col_comm; /* Коммуникатор столбца. */
    MPI_Datatype row_type; /* Тип блока в строчке. */
    MPI_Datatype col_type; /* Тип блока в колонке по все ширине поля. */
} lgca_t;


int get_nth_bit(const unsigned char value, const int n) { return (value >> n) & 1; }
void set_nth_bit(const unsigned char value, const int n, unsigned char *data) { (*data) ^= (-(unsigned char)value ^ (*data)) & (1 << n); }
int get_density(const unsigned char value);

int lgca_size(const lgca_t *l) { return l->xmax * l->ymax; }
void lgca_init(lgca_t *l);
void lgca_free(lgca_t *l);
void lgca_save_vtk(const char *path, const lgca_t *l);
void lgca_initial(lgca_t *l);
void lgca_set_value(const int i, const int j, const int direction, const unsigned char value, lgca_t *l);
void lgca_propagate(lgca_t *l);
void lgca_collide(lgca_t *l);
void lgca_bounds(lgca_t *l);
int lgca_sum(const lgca_t *l);
void decomposition(const int N, const int P, const int k, int *start, int *finish);

/* Функция пограничного обмена. */
void lgca_border_exchange(lgca_t *l);
void lgca_border_exchange_row(lgca_t *l);
void lgca_border_exchange_col(lgca_t *l);

/* Функция добавления полученного массива в изначальную сетку (l - сетка, supp - полученный массив, pos - положение (справа (1) или слева (-1))) */
void lgca_add_sub(lgca_t *l, unsigned char *sub, int pos);
void lgca_add_sub_row(lgca_t *l, unsigned char *sub, int pos);
void lgca_add_sub_col(lgca_t *l, unsigned char *sub, int pos);

/* Очищает начало или конец области (pos - начало (-1) или конец (1)), иначе идет накопление суммы */
void lgca_clear(lgca_t *l, int pos);
void lgca_clear_row(lgca_t *l, int pos);
void lgca_clear_col(lgca_t *l, int pos);

/* Запись в файл суммы для уточнения правильности распараллеливания */ 
void lgca_write_sum(const lgca_t *l, int n_frames);

/* Сбор поля у последнего процесса. */
void lgca_gather(lgca_t *l);
void lgca_gather_block(lgca_t *l);
void lgca_gather_row(lgca_t *l);
void lgca_gather_col(lgca_t *l);
void lgca_gather_row_new(lgca_t *l);
void lgca_gather_col_new(lgca_t *l);
void lgca_gather_via_mpi(lgca_t *l);

/* Проверяет, простое ли число n */
int simple(int n);

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    lgca_t lattice;
    lgca_init(&lattice);
    lgca_initial(&lattice);
    int i;
    char buf[100];
    int n_frames = 2500;
    double t = MPI_Wtime();
    for (i = 0; i < n_frames; i++) {
	if (argv[1][0] == '1') {
        if ((i % 10 == 0) && (i <= 1000)) {
            sprintf(buf, "vtk/lgca_%06d.vtk", i);
            //lgca_gather_block(&lattice);
            //lgca_gather_row(&lattice);
            //lgca_gather_col(&lattice);
            lgca_gather_via_mpi(&lattice);
            //lgca_gather_row_new(&lattice);
            //lgca_gather_col_new(&lattice);
            if (lattice.rank == lattice.num_tasks - 1) {
                lgca_save_vtk(buf, &lattice);
            }
        }
	}
	if (argv[2][0] == '1') {
        if (lattice.rank == 0) {
            printf("#%d: Sum = %d\n", i, lgca_sum(&lattice));
	    lgca_write_sum(&lattice, n_frames);
        }
	}
        lgca_propagate(&lattice);
        //lgca_border_exchange(&lattice);
        lgca_border_exchange_row(&lattice);
        lgca_border_exchange_col(&lattice);
        lgca_collide(&lattice);
        lgca_bounds(&lattice);
    }

    if (argv[3][0] == '1') {
	FILE *fp;
	char name[] = "lgca_time.txt";
	if ((fp = fopen(name, "a")) == NULL) {
		printf("Can't find file");
		return 1;
	}
	if (lattice.rank == 0) {
		t = MPI_Wtime() - t;
		fprintf(fp, "%d %f\n", lattice.num_tasks, t);
	} 
    }
    lgca_free(&lattice);
    MPI_Finalize();
    return 0;
}

void lgca_write_sum(const lgca_t *l, int n_frames) {
    FILE *fp;
    char name[] = "lgca.txt";
    if ((fp = fopen(name, "a")) == NULL) {
        printf("Can't find file");
    }
    fprintf(fp, "%d\n", lgca_sum(l));
}

int simple(int n) {
    for(int i=2;i<= n/2;i++) if( (n%i)==0 ) return 0;
    return 1;   
}

/*
void lgca_clear(lgca_t *l, int pos) {
    for (int i = 0; i < l->ymax; i++) {
        if (pos == -1) {
            l->lattice[ind(l->start - 1, i)] = 0;
        } else if (pos == 1) {
            l->lattice[ind(l->finish, i)] = 0;
        }
    }
}*/

/*
void lgca_add_sub(lgca_t *l, unsigned char *sub, int pos) {
    for (int i = 0; i < l->ymax; i++) {
        if ((get_nth_bit(sub[i], 1) == 1) && (pos == -1)) {
            set_nth_bit(1, 1, &(l->lattice[ind(l->start, i)]));
        }
        else if ((get_nth_bit(sub[i], 3) == 1) && (pos == 1)) {
            set_nth_bit(1, 3, &(l->lattice[ind(l->finish-1, i)]));
        }
    }
}*/


void lgca_clear_row(lgca_t *l, int pos) {
    for (int i = 0; i < l->finish[1] - l->start[1]; i++) {
        if (pos == -1) {
            l->lattice[ind(l->start[0] - 1, l->start[1] + i)] = 0;
        } else if (pos == 1) {
            l->lattice[ind(l->finish[0], l->start[1] + i)] = 0;
        }
    }
}

void lgca_clear_col(lgca_t *l, int pos) {
    for (int i = 0; i < l->finish[0] - l->start[0]; i++) {
        if (pos == -1) {
            l->lattice[ind(l->start[0] + i, l->start[1] - 1)] = 0;
        } else if (pos == 1) {
            l->lattice[ind(l->start[0] + i, l->finish[1])] = 0;
        }
    }    
}

void lgca_add_sub_row(lgca_t *l, unsigned char *sub, int pos) {
    for (int i = 0; i < l->finish[1] - l->start[1]; i++) {
        if ((get_nth_bit(sub[i], 1) == 1) && (pos == -1)) {
            set_nth_bit(1, 1, &(l->lattice[ind(l->start[0], l->start[1] + i)]));
        } 
        else if ((get_nth_bit(sub[i], 3) == 1) && (pos == 1)) {
            set_nth_bit(1, 3, &(l->lattice[ind(l->finish[0] - 1, l->start[1] + i)]));
        }
    }
}

void lgca_add_sub_col(lgca_t *l, unsigned char *sub, int pos) {
    for (int i = 0; i < l->finish[0] - l->start[0]; i++) {
        if ((get_nth_bit(sub[i], 0) == 1) && (pos == -1)) {
            set_nth_bit(1, 0, &(l->lattice[ind(l->start[0] + i, l->start[1])]));
        } 
        else if ((get_nth_bit(sub[i], 2) == 1) && (pos == 1)) {
            set_nth_bit(1, 2, &(l->lattice[ind(l->start[0] + i, l->finish[1] - 1)]));
        }
    }    
}

void lgca_border_exchange_row(lgca_t *l) {
    if (l->num_tasks == 1) return;

    int size = l->finish[1] - l->start[1];
    int left_rank, right_rank;

    if (l->coords[0] == 0) {
        unsigned char right[size];
        int coords_right_rank[2] = {1, l->coords[1]};
        MPI_Cart_rank(l->cart_comm, coords_right_rank, &(right_rank));

        MPI_Send(&(l->lattice[ind(l->finish[0], l->start[1])]), 1, l->col_type, right_rank, 0, l->cart_comm);
        lgca_clear_row(l, 1);
        MPI_Recv(right, size, MPI_CHAR, right_rank, 0, l->cart_comm, MPI_STATUS_IGNORE);
        lgca_add_sub_row(l, right, 1);
    } else if (l->coords[0] == l->dims[0] - 1) {
        unsigned char left[size];
        int coords_left_rank[2] = {l->dims[0] - 2, l->coords[1]};
        MPI_Cart_rank(l->cart_comm, coords_left_rank, &(left_rank));

        MPI_Send(&(l->lattice[ind(l->start[0]-1, l->start[1])]), 1, l->col_type, left_rank, 0, l->cart_comm);
        lgca_clear_row(l, -1);
        MPI_Recv(left, size, MPI_CHAR, left_rank, 0, l->cart_comm, MPI_STATUS_IGNORE);
        lgca_add_sub_row(l, left, -1);
    } else {
        unsigned char right[size];
        unsigned char left[size];
        int coords_right_rank[2] = {l->coords[0] + 1, l->coords[1]};
        int coords_left_rank[2] = {l->coords[0] - 1, l->coords[1]};
        MPI_Cart_rank(l->cart_comm, coords_right_rank, &(right_rank));
        MPI_Cart_rank(l->cart_comm, coords_left_rank, &(left_rank));

        MPI_Send(&(l->lattice[ind(l->finish[0], l->start[1])]), 1, l->col_type, right_rank, 0, l->cart_comm);
        lgca_clear_row(l, 1);
        MPI_Recv(right, size, MPI_CHAR, right_rank, 0, l->cart_comm, MPI_STATUS_IGNORE);
        lgca_add_sub_row(l, right, 1);

        MPI_Send(&(l->lattice[ind(l->start[0]-1, l->start[1])]), 1, l->col_type, left_rank, 0, l->cart_comm);
        lgca_clear_row(l, -1);
        MPI_Recv(left, size, MPI_CHAR, left_rank, 0, l->cart_comm, MPI_STATUS_IGNORE);
        lgca_add_sub_row(l, left, -1);
    }
}

void lgca_border_exchange_col(lgca_t *l) {
    if (l->num_tasks == 1) return;
    if (simple(l->num_tasks) == 1) return;

    int size = l->finish[0] - l->start[0];
    int left_rank, right_rank;

    if (l->coords[1] == 0) {
        unsigned char right[size];
        int coords_right_rank[2] = {l->coords[0], 1};
        MPI_Cart_rank(l->cart_comm, coords_right_rank, &(right_rank));

        MPI_Send(&(l->lattice[ind(l->start[0], l->finish[1])]), size, MPI_CHAR, right_rank, 0, l->cart_comm);
        lgca_clear_col(l, 1);
        MPI_Recv(right, size, MPI_CHAR, right_rank, 0, l->cart_comm, MPI_STATUS_IGNORE);
        lgca_add_sub_col(l, right, 1);
    } else if (l->coords[1] == l->dims[1] - 1) {
        unsigned char left[size];
        int coords_left_rank[2] = {l->coords[0], l->dims[1] - 2};
        MPI_Cart_rank(l->cart_comm, coords_left_rank, &(left_rank));

        MPI_Send(&(l->lattice[ind(l->start[0], l->start[1] - 1)]), size, MPI_CHAR, left_rank, 0, l->cart_comm);
        lgca_clear_col(l, -1);
        MPI_Recv(left, size, MPI_CHAR, left_rank, 0, l->cart_comm, MPI_STATUS_IGNORE);
        lgca_add_sub_col(l, left, -1);
    } else {
        unsigned char right[size];
        unsigned char left[size];
        int coords_right_rank[2] = {l->coords[0], l->coords[1] + 1};
        int coords_left_rank[2] = {l->coords[0], l->coords[1] - 1};
        MPI_Cart_rank(l->cart_comm, coords_right_rank, &(right_rank));
        MPI_Cart_rank(l->cart_comm, coords_left_rank, &(left_rank));

        MPI_Send(&(l->lattice[ind(l->start[0], l->finish[1])]), size, MPI_CHAR, right_rank, 0, l->cart_comm);
        lgca_clear_col(l, 1);
        MPI_Recv(right, size, MPI_CHAR, right_rank, 0, l->cart_comm, MPI_STATUS_IGNORE);
        lgca_add_sub_col(l, right, 1);

        MPI_Send(&(l->lattice[ind(l->start[0], l->start[1] - 1)]), size, MPI_CHAR, left_rank, 0, l->cart_comm);
        lgca_clear_col(l, -1);
        MPI_Recv(left, size, MPI_CHAR, left_rank, 0, l->cart_comm, MPI_STATUS_IGNORE);
        lgca_add_sub_col(l, left, -1);
    }    
}


/*
void lgca_border_exchange(lgca_t *l) {
    if (l->num_tasks == 1) return;

    int rank = l->rank;
    int ymax = l->ymax;
    int xmax = l->xmax;

    if (rank == 0) {
        unsigned char right[ymax];

        MPI_Send(&(l->lattice[ind(l->finish, 0)]), 1, l->column_type, 1, 0, MPI_COMM_WORLD);
        lgca_clear(l, 1);
	    MPI_Recv(right, ymax, MPI_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        lgca_add_sub(l, right, 1);
    } else if (rank == l->num_tasks - 1) {
        unsigned char left[ymax];

        MPI_Send(&(l->lattice[ind(l->start-1, 0)]), 1, l->column_type, l->num_tasks - 2, 0, MPI_COMM_WORLD);
        lgca_clear(l, -1);
	    MPI_Recv(left, ymax, MPI_CHAR, l->num_tasks - 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        lgca_add_sub(l, left, -1);
    } else {
        unsigned char right[ymax];
        unsigned char left[ymax];

        MPI_Send(&(l->lattice[ind(l->finish, 0)]), 1, l->column_type, rank + 1, 0, MPI_COMM_WORLD);
        lgca_clear(l, 1);
        MPI_Recv(right, ymax, MPI_CHAR, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        lgca_add_sub(l, right, 1);

        MPI_Send(&(l->lattice[ind(l->start-1, 0)]), 1, l->column_type, rank - 1, 0, MPI_COMM_WORLD);
        lgca_clear(l, -1);
	    MPI_Recv(left, ymax, MPI_CHAR, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        lgca_add_sub(l, left, -1);
    }
}*/



/*
void lgca_gather(lgca_t *l)
{
    if (l->rank == l->num_tasks - 1) {
        int k;
        for (k = 0; k < l->num_tasks - 1; k++) {
            int start, finish;
            decomposition(l->ymax, l->num_tasks, k, &start, &finish);
            MPI_Recv(l->lattice + ind(0, start), (finish - start) * l->xmax, MPI_CHAR, k, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    } else {
        MPI_Send(l->lattice + ind(0, l->start), (l->finish - l->start) * l->xmax, MPI_CHAR, l->num_tasks - 1, 0, MPI_COMM_WORLD);
    }
}

void lgca_gather_block(lgca_t *l)
{
    if (l->rank == l->num_tasks - 1) {
        int k;
        for (k = 0; k < l->num_tasks - 1; k++) {
            int start, finish;
            decomposition(l->xmax, l->num_tasks, k, &start, &finish);
            MPI_Recv(l->lattice + ind(start, 0), 1, l->block_type, k, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    } else {
        MPI_Send(l->lattice + ind(l->start, 0), 1, l->block_type, l->num_tasks - 1, 0, MPI_COMM_WORLD);
    }
}
*/

void lgca_gather_row(lgca_t *l)
{
    if (l->coords[0] == l->dims[0] - 1) {
        for (int k = 0; k < l->dims[0] - 1; k++) {
            int start, finish;
            decomposition(l->xmax, l->dims[0], k, &start, &finish);
            MPI_Recv(l->lattice + ind(start, l->start[1]), 1, l->row_type, k, 0, l->row_comm, MPI_STATUS_IGNORE);
        }
    } else {
        MPI_Send(l->lattice + ind(l->start[0], l->start[1]), 1, l->row_type, l->dims[0] - 1, 0, l->row_comm);    
    }
}

void lgca_gather_col(lgca_t *l) {
    if (l->coords[1] == l->dims[1] - 1) {
        int k;
        for (k = 0; k < l->dims[1] - 1; k++) {
            int start, finish;
            decomposition(l->ymax, l->dims[1], k, &start, &finish);
            MPI_Recv(l->lattice + ind(0, start), (finish - start) * l->xmax, MPI_CHAR, k, 0, l->col_comm, MPI_STATUS_IGNORE);
        }
    } else {
        MPI_Send(l->lattice + ind(0, l->start[1]), (l->finish[1] - l->start[1]) * l->xmax, MPI_CHAR, l->dims[1] - 1, 0, l->col_comm);
    }
}

/*
void lgca_gather_row_new(lgca_t *l) {
    MPI_Gather(l->lattice + ind(l->start[0], l->start[1]), 1, l->row_type, 
            l->lattice + ind(0, l->start[1]), 1, l->row_type, l->dims[0] - 1, l->row_comm);
}

void lgca_gather_col_new(lgca_t *l){
    int start, finish;
    decomposition(l->ymax, l->dims[1], 0, &start, &finish);
    MPI_Gather(l->lattice + ind(0, l->start[1]), (l->finish[1] - l->start[1]) * l->xmax, MPI_CHAR, 
            l->lattice + ind(0, 0), (finish - start) * l->xmax, MPI_CHAR, l->dims[1] - 1, l->col_comm);
}*/

void lgca_gather_via_mpi(lgca_t *l) {
    MPI_Gather(l->lattice + ind(l->start[0], l->start[1]), 1, l->row_type, 
            l->lattice + ind(0, l->start[1]), 1, l->row_type, l->dims[0] - 1, l->row_comm);
    int start, finish;
    decomposition(l->ymax, l->dims[1], 0, &start, &finish);
    MPI_Gather(l->lattice + ind(0, l->start[1]), (l->finish[1] - l->start[1]) * l->xmax, MPI_CHAR, 
            l->lattice + ind(0, 0), (finish - start) * l->xmax, MPI_CHAR, l->dims[1] - 1, l->col_comm);
}

void decomposition(const int N, const int P, const int k, int *start, int *finish)
{
    /* Условиях, что N >> P, N % P << N */
    int Nk = N / P;
    *start = k * Nk;
    *finish = *start + Nk;
    if (k == P - 1) *finish = N;
}

int get_density(const unsigned char value)
{
    int i;
    int dens = 0;
    for (i = 0; i < 4; i++) {
        dens += get_nth_bit(value, i);
    }
    return dens;
}


void lgca_initial(lgca_t *l)
{
    int ci = l->xmax / 4;
    int cj = l->ymax / 4;
    int r = l->ymax * 0.1;
    int i, j, d;
    for (j = 0; j < l->ymax; j++) {
        for (i = 0; i < l->xmax / 3; i++) {
            //for (d = 0; d < 2; d++) {
                //if (rand() % 10 < 5) {
                    lgca_set_value(i, j, 1, 1, l);
                //}
            //}
            l->lattice[ind(i,j)] = l->lattice_buf[ind(i,j)];
        }
    }
    
    for (j = 0; j < l->ymax; j++) {
        l->lattice[ind(0,j)] = WALL;
        l->lattice[ind(l->xmax-1,j)] = WALL;
    }
    for (i = 0; i < l->xmax; i++) {
        l->lattice[ind(i,0)] = WALL;
        l->lattice[ind(i,l->ymax - 1)] = WALL;
    }
    
    for (j = 0; j < l->ymax; j++) {
        for (i = l->xmax / 2; i < l->xmax / 2 + 3; i++) {
            if (j > cj + r || j < cj - r) {
                l->lattice[ind(i,j)] = WALL;
                l->lattice_buf[ind(i,j)] = WALL;
            }
        }
    }
    
    for (j = 0; j < l->ymax; j++) {
        for (i = 0; i < l->xmax; i++) {
            l->lattice_buf[ind(i,j)] = l->lattice[ind(i,j)];
        }
    }
}

int lgca_sum(const lgca_t *l)
{
    int i, j, d;
    int cnt = 0;
    for (j = l->start[1]; j < l->finish[1]; j++) {
        for (i = 0; i < l->xmax; i++) {
            unsigned char v = l->lattice[ind(i,j)];
            for (d = 0; d < 4; d++) {
                cnt += get_nth_bit(v, d);
            }
        }
    }
    return cnt;
}    

void lgca_collide(lgca_t *l)
{
    int i, j;
    int col_cnt = 0;
    for (j = l->start[1]; j < l->finish[1]; j++) {
        for (i = l->start[0]; i < l->finish[0]; i++) {
            unsigned char v = l->lattice[ind(i,j)];
            if (v == 10) { l->lattice[ind(i,j)] = 5; col_cnt++; }
            if (v ==  5) { l->lattice[ind(i,j)] = 10; col_cnt++; }
            
            /* Для визуальной проверки сохранения и декомпозиции. */
            //l->lattice[ind(i,j)] = l->rank;
        }
    }
    //printf("collisions = %d\n", col_cnt);
}

void lgca_bounds(lgca_t *l)
{
    int i, j;
    for (j = l->start[1]; j < l->finish[1]; j++) {
        for (i = l->start[0]; i < l->finish[0]; i++) {
            unsigned char v = l->lattice[ind(i,j)];
            if (v == WALL) continue;
            if (get_nth_bit(v, 0) && l->lattice[ind(i,j+1)] == WALL) lgca_set_value(i, j, 2, 1, l);
            if (get_nth_bit(v, 1) && l->lattice[ind(i+1,j)] == WALL) lgca_set_value(i, j, 3, 1, l);
            if (get_nth_bit(v, 2) && l->lattice[ind(i,j-1)] == WALL) lgca_set_value(i, j, 0, 1, l);
            if (get_nth_bit(v, 3) && l->lattice[ind(i-1,j)] == WALL) lgca_set_value(i, j, 1, 1, l);
        }
    }
}


void lgca_propagate(lgca_t *l)
{
    int i, j;
    for (j = l->start[1]; j < l->finish[1]; j++) {
        for (i = l->start[0]; i < l->finish[0]; i++) {
            unsigned char v = l->lattice[ind(i,j)];
            if (v == WALL) continue;
            if (get_nth_bit(v, 0) && l->lattice[ind(i,j+1)] != WALL) {
                lgca_set_value(i, j+1, 0, 1, l);
            }
            if (get_nth_bit(v, 1) && l->lattice[ind(i+1,j)] != WALL) {
                lgca_set_value(i+1, j, 1, 1, l);
            }
            if (get_nth_bit(v, 2) && l->lattice[ind(i,j-1)] != WALL) {
                lgca_set_value(i, j-1, 2, 1, l);
            }
            if (get_nth_bit(v, 3) && l->lattice[ind(i-1,j)] != WALL) {
                lgca_set_value(i-1, j, 3, 1, l);
            }
            l->lattice[ind(i,j)] = 0;
        }
    }
    unsigned char *t = l->lattice;
    l->lattice = l->lattice_buf;
    l->lattice_buf = t;
}

void lgca_set_value(const int i, const int j, const int direction, const unsigned char value, lgca_t *l)
{
    set_nth_bit(value, direction, l->lattice_buf + ind(i, j));
}

void lgca_init(lgca_t *l)
{
    l->xmax = 100;
    l->ymax = 100;
    l->lattice = calloc(lgca_size(l), sizeof(unsigned char));
    l->lattice_buf = calloc(lgca_size(l), sizeof(unsigned char));
    
    
    /* Этап декомпозиции. */
    MPI_Comm_size(MPI_COMM_WORLD, &(l->num_tasks));
    MPI_Comm_rank(MPI_COMM_WORLD, &(l->rank));
    
    l->dims[0] = l->dims[1] = 0;
    MPI_Dims_create(l->num_tasks, 2, l->dims);

    int periods[2] = {0, 0};
    MPI_Cart_create(MPI_COMM_WORLD, 2, l->dims, periods, 0, &(l->cart_comm));
    MPI_Cart_coords(l->cart_comm, l->rank, 2, l->coords);
    decomposition(l->xmax, l->dims[0], l->coords[0], l->start, l->finish);
    decomposition(l->ymax, l->dims[1], l->coords[1], l->start+1, l->finish+1);

    int start, finish;
    decomposition(l->xmax, l->dims[0], 0, &start, &finish);

    MPI_Datatype oldtype;
    MPI_Type_vector(l->finish[1] - l->start[1], finish-start, l->xmax, MPI_CHAR,
                    &(oldtype));
    MPI_Aint lb;
    MPI_Aint extent;
    //MPI_Type_lb(oldtype, &lb);
    //MPI_Type_extent(oldtype, &extent);
    MPI_Type_get_extent(oldtype, &lb, &extent);
    MPI_Type_create_resized(oldtype, lb, finish - start, &(l->row_type));
    MPI_Type_commit(&(l->row_type));

    MPI_Type_vector(l->finish[1] - l->start[1], 1, l->xmax, MPI_CHAR, &(l->col_type));
    MPI_Type_commit(&(l->col_type));

    MPI_Comm_split(l->cart_comm, l->coords[1], l->coords[0], &(l->row_comm));
    MPI_Comm_split(l->cart_comm, l->coords[0], l->coords[1], &(l->col_comm));

    MPI_Type_free(&(oldtype));
}

void lgca_free(lgca_t *l)
{
    l->xmax = 0;
    l->ymax = 0;
    free(l->lattice);
    free(l->lattice_buf);
    //MPI_Type_free(&(l->block_type));
    //MPI_Type_free(&(l->column_type));
    MPI_Type_free(&(l->row_type));
    MPI_Type_free(&(l->col_type));
}

void lgca_save_vtk(const char *path, const lgca_t *l)
{
	FILE *f;
	int i1, i2;
	f = fopen(path, "w");
	assert(f);
	fprintf(f, "# vtk DataFile Version 3.0\n");
	fprintf(f, "Created by lgca_save_vtk\n");
	fprintf(f, "ASCII\n");
	fprintf(f, "DATASET STRUCTURED_POINTS\n");
	fprintf(f, "DIMENSIONS %d %d 1\n", l->xmax+1, l->ymax+1);
	fprintf(f, "SPACING %d %d 0.0\n", 1, 1);
	fprintf(f, "ORIGIN %d %d 0.0\n", 0, 0);
	fprintf(f, "CELL_DATA %d\n", lgca_size(l));
	
	fprintf(f, "SCALARS value char 1\n");
	fprintf(f, "LOOKUP_TABLE value_table\n");
	for (i2 = 0; i2 < l->ymax; i2++) {
		for (i1 = 0; i1 < l->xmax; i1++) {
			fprintf(f, "%d\n", l->lattice[ind(i1, i2)]);
		}
	}
	fprintf(f, "SCALARS density char 1\n");
	fprintf(f, "LOOKUP_TABLE density_table\n");
	for (i2 = 0; i2 < l->ymax; i2++) {
		for (i1 = 0; i1 < l->xmax; i1++) {
			fprintf(f, "%d\n", get_density(l->lattice[ind(i1, i2)]));
		}
	}
	fclose(f);
}
