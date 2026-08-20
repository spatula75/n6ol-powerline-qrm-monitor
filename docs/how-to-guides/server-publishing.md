# Publish to a web server

## Introduction

If you run your own web site and want to automatically publish your charts and data, this guide is for you.

Publishing is performed once per minute after the current CSV file has been updated and the charts regenerated.  It is performed using SFTP using a key you create and specify to access your web host.

## Setup

### Create your SSH key

First you need to generate an SSH key pair you can use to access your web host.  On a machine with ssh installed:

`ssh-keygen -t ed25519 -f ~/.buzz/buzz.pem -N ""`

This generates a random SSH key and places it in the `.buzz` directory under your home directory, names it `buzz.pem` and configures it to work without a passphrase (required for automation).  It also creates a `buzz.pem.pub` file.

Next, log in to your web host and navigate to the `.ssh` directory off your home directory.  Look for a file called `authorized_keys` or create it if it does not exist. To this file, append the contents of the `buzz.pem.pub` file.  You can also do this in one step if you like:

`cat ~/.buzz/buzz.pem.pub | ssh yourname@yourserver "cat >> ~/.ssh/authorized_keys"`

Now you can test and make sure you can access your remote server:

`ssh -i ~/.buzz/buzz.pem yourname@yourserver`

If you find yourself logged in, you're ready for the next step.

### Configure your key and enable publishing

Launch the setup program either by running `setup.bat` or `./setup.sh`, and navigate to "Web Publishing".  If not already enabled, select "Publish to a web server", use the arrow keys to move to "On", press `SPACE` or `ENTER` to select it, then tab to `OK` and press `ENTER` to enable publishing.  The rest of the publishing settings appear once it's on.

Fill in your server hostname and username.  For the "Remote path" enter a full directory path on your web server where files are expected to be read by your web server;  for example, `/var/www/html/noise`. This directory must already exist on your web host, so make sure you create it if necessary.

Your ssh key location should already be configured to point at `buzz.pem` from above, but if you placed it in a different location, update the path to the key here as well.

Then press `ESC` to go back to the main configuration screen and select `FINISH`.

Lastly, restart your monitor if it was already running.
