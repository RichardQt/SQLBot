from apps.template.template import get_activity_extract_template_file


def get_activity_extract_template():
    template = get_activity_extract_template_file()
    return template['template']['activity_extract']
