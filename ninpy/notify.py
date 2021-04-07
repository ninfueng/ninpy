"""Basic Notify script.
"""
import os
from knockknock import desktop_sender, slack_sender
from .crypto import decrypt
ENCRYPTED_SLACK_WEBHOOK = 'gAAAAABgbT7--YUvDqt93ONtkIlMHNRRTOx0pXa5RIbGVJqc_7N74mY0erv8VbLKCC020xMIQwlzv4R_RwPBtwdhs7tdZ7-K8e7SmHt7lLZB3MV_JBkTKz6hFpM8eDMCvU5JO7qfykm-gu0M_eINQV_4cXrkBX162f-4wLmDnPti6RhURFz0yvjuG4r9AXueeRt8l_6kcIcG'

try:
    key = os.environ['SLACK_KEY']
    SLACK_WEBHOOK = decrypt(ENCRYPTED_SLACK_WEBHOOK.encode(), key.encode())
except KeyError:
    raise KeyError(
        f'Environment variable `SLACK_KEY` is not found.'
        f'Please put `export SLACK_KEY="KEY_VALUE".')


@desktop_sender(title='Training was done!!!')
@slack_sender(
    webhook_url=SLACK_WEBHOOK, channel='tracking-training', user_mentions=['@binmanager'])
def basic_notify(results_dict: dict) -> dict:
    """Basic wrapper function to notify to both slack and desktop.
    """
    assert isinstance(results_dict, dict)
    return results_dict


if __name__ == '__main__':
    basic_notify({})