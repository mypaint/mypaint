
#include <mypaint-brush.h>

#include <stdio.h>
#include <malloc.h>
#include <string.h>

char *
read_file(char *path)
{
    long file_size = -1L;
    FILE *file = fopen(path, "r");

    fseek(file , 0 , SEEK_END);
    file_size = ftell(file);
    rewind(file);

    char *buffer = (char *)malloc(sizeof(char)*file_size);
    size_t result = fread(buffer, 1, file_size, file);

    fclose(file);

    if (!result) {
        free(buffer);
        return NULL;
    }
    return buffer;
}

int
expect_int(int expected, int actual, const char *description) {
    int passed = 1;
    if (expected != actual) {
        passed = 0;
        fprintf(stderr, "%s: Expected %d, got %d\n", description, expected, actual);
    }
    return passed;
}

int
expect_float(float expected, float actual, const char *description) {
    int passed = 1;
    if (expected != actual) {
        passed = 0;
        fprintf(stderr, "%s: Expected %f, got %f\n", description, expected, actual);
    }

    return passed;
}


typedef struct {
    const char *cname;
    float base_value;
} BaseValue;

int
test_brush_load_base_values(void)
{
    char *input_json = read_file("brushes/impressionism.myb");

    BaseValue expected_base_values[] = {
        {"anti_aliasing", 0.66},
        {"change_color_h",     0.0},
        {"change_color_hsl_s",     0.0},
        {"change_color_hsv_s",     0.0},
        {"change_color_l",     0.0},
        {"change_color_v",     0.0},
        {"color_h",     0.0},
        {"color_s",     0.0},
        {"color_v",     0.0},
        {"colorize",     0.0},
        {"custom_input",     0.0},
        {"custom_input_slowness",     0.0},
        {"dabs_per_actual_radius",     6.0},
        {"dabs_per_basic_radius",     6.0},
        {"dabs_per_second",     80.0},
        {"direction_filter",     2.0},
        {"elliptical_dab_angle",     0.0},
        {"elliptical_dab_ratio",     7.1},
        {"eraser",     0.0},
        {"hardness",     0.8},
        {"lock_alpha",     0.0},
        {"offset_by_random",     0.6},
        {"offset_by_speed",     0.0},
        {"offset_by_speed_slowness",     1.0},
        {"opaque",     1.0},
        {"opaque_linearize",     0.9},
        {"opaque_multiply",     0.0},
        {"radius_by_random",     0.0},
        {"radius_logarithmic",     2.0},
        {"restore_color",     0.0},
        {"slow_tracking", 0.0},
        {"slow_tracking_per_dab",     0.0},
        {"smudge",     0.9},
        {"smudge_length",     0.0},
        {"smudge_radius_log",     0.0},
        {"speed1_gamma",     4.0},
        {"speed1_slowness",     0.04},
        {"speed2_gamma",     4.0},
        {"speed2_slowness",     0.8},
        {"stroke_duration_logarithmic",     6.0},
        {"stroke_holdtime",     10.0},
        {"stroke_threshold",     0.0},
        {"tracking_noise",     0.2}
    };
    int number_of_expected_base_values = sizeof(expected_base_values) / sizeof(BaseValue);

    if (!number_of_expected_base_values == MYPAINT_BRUSH_SETTINGS_COUNT) {
        // Would happen if new settings are introduced. Should update the test then
        fprintf(stderr, "Warning: number of base values tested does not match number of settings.");
    }

    MyPaintBrush *brush = mypaint_brush_new();
    mypaint_brush_from_string(brush, input_json);

    int passed = 1;

    // Check base values
    for (int i=0; i<number_of_expected_base_values; i++) {
        const BaseValue *base_value = &expected_base_values[i];
        int correct = -1;

        MyPaintBrushSetting id = mypaint_brush_setting_from_cname(base_value->cname);
        float expected = base_value->base_value;
        float actual = mypaint_brush_get_base_value(brush, id);
        correct = (expected == actual);

        if (correct != 1) {
            fprintf(stderr, "Wrong base value for %s: Expected %f, got %f\n", base_value->cname, expected, actual);
            passed = 0;
        }
    }


    mypaint_brush_destroy(brush);

    return passed;
}

typedef struct {
    const char *cname;
    int no_mapping_points;
    float *mapping_points;
} Input;

typedef struct {
    const char *cname;
    int no_inputs;
    Input *inputs;
} Inputs;

int
test_brush_load_inputs(void)
{
    char *input_json = read_file("brushes/modelling.myb");


    float opaque_speed2[] = {0.0, 0.28, 0.518519, 0.032083, 1.888889, -0.16625, 4.0, -0.28};
    Input opaque_setting_inputs[1];
    opaque_setting_inputs[0].cname = "speed2";
    opaque_setting_inputs[0].mapping_points = opaque_speed2;
    opaque_setting_inputs[0].no_mapping_points = 4;

    float opaque_multiply_pressure[] = { 0.0, 0.0, 1.0, 0.52};
    Input opaque_multiply_setting_inputs[1];
    opaque_multiply_setting_inputs[0].cname = "pressure";
    opaque_multiply_setting_inputs[0].mapping_points = opaque_multiply_pressure;
    opaque_multiply_setting_inputs[0].no_mapping_points = 2;

    float radius_logarithmic_pressure[] = {0.0, 0.326667, 1.0, -0.49};
    float radius_logarithmic_speed1[] = {0.0, 0.0, 1.0, 0.75};
    float radius_logarithmic_speed2[] = { 0.0, -0.15, 4.0, 1.05};
    Input radius_logarithmic_setting_inputs[] = {
        {"pressure", 2, radius_logarithmic_pressure},
        {"speed1", 2, radius_logarithmic_speed1},
        {"speed2", 2, radius_logarithmic_speed2}
    };

    float smudge_pressure[] = {0.0, -0.0, 0.290123, -0.0375, 0.645062, -0.15, 1.0, -0.4};
    Input smudge_setting_inputs[] = {
            {"pressure", 4, smudge_pressure}
    };

    Inputs expected_inputs[] = {
        {"anti_aliasing", 0, NULL},
        {"change_color_h", 0,     NULL},
        {"change_color_hsl_s", 0,     NULL},
        {"change_color_hsv_s", 0,     NULL},
        {"change_color_l", 0,     NULL},
        {"change_color_v", 0,     NULL},
        {"color_h", 0,     NULL},
        {"color_s", 0,     NULL},
        {"color_v", 0,     NULL},
        {"colorize", 0,     NULL},
        {"custom_input", 0,     NULL},
        {"custom_input_slowness", 0,     NULL},
        {"dabs_per_actual_radius", 0,     NULL},
        {"dabs_per_basic_radius", 0,     NULL},
        {"dabs_per_second", 0,     NULL},
        {"direction_filter", 0,     NULL},
        {"elliptical_dab_angle", 0,     NULL},
        {"elliptical_dab_ratio", 0,     NULL},
        {"eraser", 0,     NULL},
        {"hardness", 0,     NULL},
        {"lock_alpha", 0,     NULL},
        {"offset_by_random", 0,     NULL},
        {"offset_by_speed", 0,     NULL},
        {"offset_by_speed_slowness", 0,     NULL},
        {"opaque", 1,    opaque_setting_inputs},
        {"opaque_linearize", 0,     NULL},
        {"opaque_multiply",  1,    opaque_multiply_setting_inputs},
        {"radius_by_random", 0,     NULL},
        {"radius_logarithmic", 3,     radius_logarithmic_setting_inputs},
        {"restore_color", 0,     NULL},
        {"slow_tracking", 0, NULL},
        {"slow_tracking_per_dab", 0,     NULL},
        {"smudge", 1,    smudge_setting_inputs},
        {"smudge_length", 0,     NULL},
        {"smudge_radius_log", 0,     NULL},
        {"speed1_gamma", 0,     NULL},
        {"speed1_slowness", 0,     NULL},
        {"speed2_gamma", 0,     NULL},
        {"speed2_slowness", 0,     NULL},
        {"stroke_duration_logarithmic", 0,     NULL},
        {"stroke_holdtime", 0,     NULL},
        {"stroke_threshold",  0,    NULL},
        {"tracking_noise",  0,    NULL}
    };
    int number_of_expected_inputs = sizeof(expected_inputs) / sizeof(Inputs);

    if (!number_of_expected_inputs == MYPAINT_BRUSH_SETTINGS_COUNT) {
        // Would happen if new settings are introduced. Should update the test then
        fprintf(stderr, "Warning: number of values tested does not match number of settings.");
    }

    MyPaintBrush *brush = mypaint_brush_new();
    mypaint_brush_from_string(brush, input_json);

    int passed = 1;

    // Check input/dynamics values
    for (int i=0; i<number_of_expected_inputs; i++) {
        const Inputs *inputs = &expected_inputs[i];
        const char *setting_name = inputs->cname;
        MyPaintBrushSetting setting_id = mypaint_brush_setting_from_cname(inputs->cname);
        int correct = 1;

        if (inputs->inputs == NULL) {
            if (!mypaint_brush_is_constant(brush, setting_id)) {
                fprintf(stderr, "Expected setting %s to be constant (have 0 inputs)", setting_name);
                correct = 0;
            }

        } else {

            int number_of_inputs = inputs->no_inputs;
            expect_int(number_of_inputs, mypaint_brush_get_inputs_used_n(brush, setting_id), "");

            for (int i=0; i<number_of_inputs; i++) {
                const Input *input = &(inputs->inputs[i]);
                //const char *input_name = input->cname;
                MyPaintBrushInput input_id = mypaint_brush_input_from_cname(input->cname);

                int expected_number_of_mapping_points = input->no_mapping_points;
                int actual_number_of_mapping_points = mypaint_brush_get_mapping_n(brush, setting_id, input_id);

                if (!expect_int(expected_number_of_mapping_points, actual_number_of_mapping_points, "Mapping points. ")) {
                    correct = 0;
                }

                for (int i=0; i<expected_number_of_mapping_points; i++) {
                    float expected_x = input->mapping_points[i*2];
                    float expected_y = input->mapping_points[(i*2)+1];

                    float actual_x;
                    float actual_y;
                    mypaint_brush_get_mapping_point(brush, setting_id, input_id, i, &actual_x, &actual_y);

                    if (!expect_float(expected_x, actual_x, ""))
                        correct = 0;
                    if (!expect_float(expected_y, actual_y, ""))
                        correct = 0;
                }
            }

        }

        if (correct != 1) {
            passed = 0;
        }
    }


    mypaint_brush_destroy(brush);

    return passed;
}

typedef int (*TestFunction) (void);

typedef struct {
    char *id;
    TestFunction function;
} TestCase;

int
main(int argc, char **argv)
{
    static const char * const pass = "PASS";
    static const char * const fail = "FAIL";

    TestCase test_cases[] = {
        {"/brush/persistence/load/base_values", test_brush_load_base_values},
        {"/brush/persistence/load/inputs", test_brush_load_inputs},
    };
    const int no_test_cases = sizeof(test_cases) / sizeof(TestCase);

    int failures = 0;
    for (int i=0; i<no_test_cases; i++) {
        const TestCase *test_case = &test_cases[i];
        int result = test_case->function();
        if (result != 1); failures++;
        fprintf(stdout, "%s: %s\n", test_case->id, (result == 1) ? pass : fail);
        fflush(stdout);
    }

    return (failures != 0);
}
