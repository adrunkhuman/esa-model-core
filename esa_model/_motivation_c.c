#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#include <numpy/arrayobject.h>

static PyObject *simulate_conditioned_zone_counts(PyObject *self, PyObject *args) {
    PyObject *starting_obj, *samples_obj, *noise_obj, *homes_obj, *aways_obj;
    PyObject *p_home_obj, *p_draw_obj, *conditioned_obj, *members_obj;
    PyObject *offsets_obj, *zones_obj, *requested_obj;
    PyArrayObject *starting = NULL, *samples = NULL, *noise = NULL;
    PyArrayObject *homes = NULL, *aways = NULL, *p_home = NULL, *p_draw = NULL;
    PyArrayObject *conditioned = NULL, *members = NULL, *offsets = NULL;
    PyArrayObject *zones = NULL, *requested = NULL, *counts = NULL;
    npy_intp *team_groups = NULL;
    npy_intp *requested_slots = NULL;
    npy_intp *rank_order = NULL;
    npy_int16 *base_points = NULL;
    double *team_scores = NULL;

    if (!PyArg_ParseTuple(
            args,
            "OOOOOOOOOOOO:simulate_conditioned_zone_counts",
            &starting_obj,
            &samples_obj,
            &noise_obj,
            &homes_obj,
            &aways_obj,
            &p_home_obj,
            &p_draw_obj,
            &conditioned_obj,
            &members_obj,
            &offsets_obj,
            &zones_obj,
            &requested_obj)) {
        return NULL;
    }

    starting = (PyArrayObject *)PyArray_FROM_OTF(starting_obj, NPY_INT16, NPY_ARRAY_IN_ARRAY);
    samples = (PyArrayObject *)PyArray_FROM_OTF(samples_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    noise = (PyArrayObject *)PyArray_FROM_OTF(noise_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    homes = (PyArrayObject *)PyArray_FROM_OTF(homes_obj, NPY_INTP, NPY_ARRAY_IN_ARRAY);
    aways = (PyArrayObject *)PyArray_FROM_OTF(aways_obj, NPY_INTP, NPY_ARRAY_IN_ARRAY);
    p_home = (PyArrayObject *)PyArray_FROM_OTF(p_home_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    p_draw = (PyArrayObject *)PyArray_FROM_OTF(p_draw_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    conditioned = (PyArrayObject *)PyArray_FROM_OTF(conditioned_obj, NPY_INTP, NPY_ARRAY_IN_ARRAY);
    members = (PyArrayObject *)PyArray_FROM_OTF(members_obj, NPY_INTP, NPY_ARRAY_IN_ARRAY);
    offsets = (PyArrayObject *)PyArray_FROM_OTF(offsets_obj, NPY_INTP, NPY_ARRAY_IN_ARRAY);
    zones = (PyArrayObject *)PyArray_FROM_OTF(zones_obj, NPY_INTP, NPY_ARRAY_IN_ARRAY);
    requested = (PyArrayObject *)PyArray_FROM_OTF(requested_obj, NPY_INTP, NPY_ARRAY_IN_ARRAY);
    if (!starting || !samples || !noise || !homes || !aways || !p_home || !p_draw ||
        !conditioned || !members || !offsets || !zones || !requested) {
        goto fail;
    }

    if (PyArray_NDIM(starting) != 1 || PyArray_NDIM(samples) != 2 || PyArray_NDIM(noise) != 2 ||
        PyArray_NDIM(homes) != 1 || PyArray_NDIM(aways) != 1 || PyArray_NDIM(p_home) != 1 ||
        PyArray_NDIM(p_draw) != 1 || PyArray_NDIM(conditioned) != 1 || PyArray_NDIM(members) != 1 ||
        PyArray_NDIM(offsets) != 1 || PyArray_NDIM(zones) != 1 || PyArray_NDIM(requested) != 1) {
        PyErr_SetString(PyExc_ValueError, "conditioned simulation arrays have invalid dimensions");
        goto fail;
    }

    const npy_intp teams = PyArray_DIM(starting, 0);
    const npy_intp fixtures = PyArray_DIM(samples, 0);
    const npy_intp simulations = PyArray_DIM(samples, 1);
    const npy_intp conditions = PyArray_DIM(conditioned, 0);
    const npy_intp requested_count = PyArray_DIM(requested, 0);
    const npy_intp group_count = PyArray_DIM(offsets, 0) - 1;
    if (teams < 1 || simulations < 1 || group_count < 1 ||
        PyArray_DIM(noise, 0) != simulations || PyArray_DIM(noise, 1) != teams ||
        PyArray_DIM(homes, 0) != fixtures || PyArray_DIM(aways, 0) != fixtures ||
        PyArray_DIM(p_home, 0) != fixtures || PyArray_DIM(p_draw, 0) != fixtures ||
        PyArray_DIM(members, 0) != teams || PyArray_DIM(zones, 0) != teams) {
        PyErr_SetString(PyExc_ValueError, "conditioned simulation array shapes are inconsistent");
        goto fail;
    }

    const npy_intp *home_data = (const npy_intp *)PyArray_DATA(homes);
    const npy_intp *away_data = (const npy_intp *)PyArray_DATA(aways);
    const npy_intp *conditioned_data = (const npy_intp *)PyArray_DATA(conditioned);
    const npy_intp *member_data = (const npy_intp *)PyArray_DATA(members);
    const npy_intp *offset_data = (const npy_intp *)PyArray_DATA(offsets);
    const npy_intp *zone_data = (const npy_intp *)PyArray_DATA(zones);
    const npy_intp *requested_data = (const npy_intp *)PyArray_DATA(requested);
    if (offset_data[0] != 0 || offset_data[group_count] != teams) {
        PyErr_SetString(PyExc_ValueError, "rank-group offsets must cover every team");
        goto fail;
    }

    team_groups = PyMem_Malloc((size_t)teams * sizeof(npy_intp));
    if (!team_groups) {
        PyErr_NoMemory();
        goto fail;
    }
    for (npy_intp team = 0; team < teams; team++) {
        team_groups[team] = -1;
    }
    for (npy_intp group = 0; group < group_count; group++) {
        if (offset_data[group] > offset_data[group + 1]) {
            PyErr_SetString(PyExc_ValueError, "rank-group offsets must be ordered");
            goto fail;
        }
        for (npy_intp member = offset_data[group]; member < offset_data[group + 1]; member++) {
            const npy_intp team = member_data[member];
            if (team < 0 || team >= teams || team_groups[team] != -1) {
                PyErr_SetString(PyExc_ValueError, "rank groups must partition the team indices");
                goto fail;
            }
            team_groups[team] = group;
        }
    }
    for (npy_intp team = 0; team < teams; team++) {
        if (team_groups[team] == -1 || zone_data[team] < 0 || zone_data[team] >= 5) {
            PyErr_SetString(PyExc_ValueError, "rank groups or zones contain an invalid index");
            goto fail;
        }
    }
    for (npy_intp fixture = 0; fixture < fixtures; fixture++) {
        if (home_data[fixture] < 0 || home_data[fixture] >= teams ||
            away_data[fixture] < 0 || away_data[fixture] >= teams) {
            PyErr_SetString(PyExc_ValueError, "fixture teams contain an invalid index");
            goto fail;
        }
    }
    for (npy_intp condition = 0; condition < conditions; condition++) {
        if (conditioned_data[condition] < 0 || conditioned_data[condition] >= fixtures) {
            PyErr_SetString(PyExc_ValueError, "conditioned fixtures contain an invalid index");
            goto fail;
        }
    }
    for (npy_intp index = 0; index < requested_count; index++) {
        if (requested_data[index] < 0 || requested_data[index] >= teams) {
            PyErr_SetString(PyExc_ValueError, "requested teams contain an invalid index");
            goto fail;
        }
    }

    requested_slots = PyMem_Malloc((size_t)teams * sizeof(npy_intp));
    rank_order = PyMem_Malloc((size_t)teams * sizeof(npy_intp));
    team_scores = PyMem_Malloc((size_t)teams * sizeof(double));
    if (!requested_slots || !rank_order || !team_scores) {
        PyErr_NoMemory();
        goto fail;
    }
    for (npy_intp team = 0; team < teams; team++) {
        requested_slots[team] = -1;
    }
    for (npy_intp index = 0; index < requested_count; index++) {
        requested_slots[requested_data[index]] = index;
    }

    if (simulations > NPY_MAX_INTP / teams) {
        PyErr_NoMemory();
        goto fail;
    }
    base_points = PyMem_Malloc((size_t)(simulations * teams) * sizeof(npy_int16));
    if (!base_points) {
        PyErr_NoMemory();
        goto fail;
    }

    npy_intp output_shape[3] = {conditions * 2, requested_count, 5};
    counts = (PyArrayObject *)PyArray_ZEROS(3, output_shape, NPY_INT64, 0);
    if (!counts) {
        goto fail;
    }

    const npy_int16 *starting_data = (const npy_int16 *)PyArray_DATA(starting);
    const double *sample_data = (const double *)PyArray_DATA(samples);
    const double *noise_data = (const double *)PyArray_DATA(noise);
    const double *p_home_data = (const double *)PyArray_DATA(p_home);
    const double *p_draw_data = (const double *)PyArray_DATA(p_draw);
    npy_int64 *count_data = (npy_int64 *)PyArray_DATA(counts);

    Py_BEGIN_ALLOW_THREADS
    for (npy_intp simulation = 0; simulation < simulations; simulation++) {
        for (npy_intp team = 0; team < teams; team++) {
            base_points[simulation * teams + team] = starting_data[team];
        }
    }
    for (npy_intp fixture = 0; fixture < fixtures; fixture++) {
        const npy_intp home = home_data[fixture];
        const npy_intp away = away_data[fixture];
        const double draw_cutoff = p_home_data[fixture] + p_draw_data[fixture];
        for (npy_intp simulation = 0; simulation < simulations; simulation++) {
            const double sample = sample_data[fixture * simulations + simulation];
            if (sample < p_home_data[fixture]) {
                base_points[simulation * teams + home] += 3;
            } else if (sample < draw_cutoff) {
                base_points[simulation * teams + home] += 1;
                base_points[simulation * teams + away] += 1;
            } else {
                base_points[simulation * teams + away] += 3;
            }
        }
    }

    for (npy_intp condition = 0; condition < conditions; condition++) {
        const npy_intp fixture = conditioned_data[condition];
        const npy_intp home = home_data[fixture];
        const npy_intp away = away_data[fixture];
        const double draw_cutoff = p_home_data[fixture] + p_draw_data[fixture];
        for (npy_intp forced_away = 0; forced_away < 2; forced_away++) {
            const npy_intp scenario = condition * 2 + forced_away;
            for (npy_intp simulation = 0; simulation < simulations; simulation++) {
                const double sample = sample_data[fixture * simulations + simulation];
                npy_int16 old_home = 0, old_away = 3;
                if (sample < p_home_data[fixture]) {
                    old_home = 3;
                    old_away = 0;
                } else if (sample < draw_cutoff) {
                    old_home = 1;
                    old_away = 1;
                }
                const npy_int16 home_delta = (forced_away == 0 ? 3 : 0) - old_home;
                const npy_int16 away_delta = (forced_away == 0 ? 0 : 3) - old_away;
                const npy_intp point_base = simulation * teams;
                const npy_intp noise_base = simulation * teams;
                for (npy_intp team = 0; team < teams; team++) {
                    const npy_int16 team_delta =
                        (team == home ? home_delta : 0) + (team == away ? away_delta : 0);
                    team_scores[team] = base_points[point_base + team] + team_delta + noise_data[noise_base + team];
                }
                for (npy_intp group = 0; group < group_count; group++) {
                    const npy_intp start = offset_data[group];
                    const npy_intp end = offset_data[group + 1];
                    for (npy_intp position = start; position < end; position++) {
                        const npy_intp team = member_data[position];
                        npy_intp insertion = position;
                        while (insertion > start && team_scores[rank_order[insertion - 1]] < team_scores[team]) {
                            rank_order[insertion] = rank_order[insertion - 1];
                            insertion--;
                        }
                        rank_order[insertion] = team;
                    }
                    for (npy_intp position = start; position < end; position++) {
                        const npy_intp requested_index = requested_slots[rank_order[position]];
                        if (requested_index >= 0) {
                            const npy_intp zone = zone_data[position];
                            count_data[(scenario * requested_count + requested_index) * 5 + zone]++;
                        }
                    }
                }
            }
        }
    }
    Py_END_ALLOW_THREADS

    Py_DECREF(starting);
    Py_DECREF(samples);
    Py_DECREF(noise);
    Py_DECREF(homes);
    Py_DECREF(aways);
    Py_DECREF(p_home);
    Py_DECREF(p_draw);
    Py_DECREF(conditioned);
    Py_DECREF(members);
    Py_DECREF(offsets);
    Py_DECREF(zones);
    Py_DECREF(requested);
    PyMem_Free(team_groups);
    PyMem_Free(requested_slots);
    PyMem_Free(rank_order);
    PyMem_Free(base_points);
    PyMem_Free(team_scores);
    return (PyObject *)counts;

fail:
    Py_XDECREF(starting);
    Py_XDECREF(samples);
    Py_XDECREF(noise);
    Py_XDECREF(homes);
    Py_XDECREF(aways);
    Py_XDECREF(p_home);
    Py_XDECREF(p_draw);
    Py_XDECREF(conditioned);
    Py_XDECREF(members);
    Py_XDECREF(offsets);
    Py_XDECREF(zones);
    Py_XDECREF(requested);
    Py_XDECREF(counts);
    PyMem_Free(team_groups);
    PyMem_Free(requested_slots);
    PyMem_Free(rank_order);
    PyMem_Free(base_points);
    PyMem_Free(team_scores);
    return NULL;
}

static PyMethodDef motivation_methods[] = {
    {"simulate_conditioned_zone_counts", simulate_conditioned_zone_counts, METH_VARARGS,
     "Simulate conditioned objective-zone counts for requested teams."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef motivation_module = {
    PyModuleDef_HEAD_INIT,
    "_motivation_c",
    NULL,
    -1,
    motivation_methods,
};

PyMODINIT_FUNC PyInit__motivation_c(void) {
    import_array();
    return PyModule_Create(&motivation_module);
}
