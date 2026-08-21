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

Fill in your server hostname and username.  For the "Remote path" enter a full directory path on your web server where files are expected to be read by your web server;  for example, `/var/www/html/noise/`.  This directory must already exist on your web host, so make sure you create it if necessary.  End the path with a slash.  If you leave it off, the monitor adds it for you.

Your ssh key location should already be configured to point at `buzz.pem` from above, but if you placed it in a different location, update the path to the key here as well.

Then press `ESC` to go back to the main configuration screen and select `FINISH`.

### Choose how the page finds today's chart

Your charts are named by date, so `noise_plot_movavg.2026-08-21.png` becomes a different file tomorrow.  The published page doesn't need to know about any of that.  It reads one fixed address instead, `data/current.png`, and the monitor keeps that address pointing at the current day's chart for you.

There are two ways the monitor can do that, and the "Current chart link" setting chooses between them.

**Copy** is the default and works everywhere.  Each cycle the monitor uploads the chart a second time under the fixed name.  That costs one extra upload per minute, around 85 kB, which almost nobody will notice.

**Symlink** saves that upload by pointing the fixed name at the dated chart instead.  Your web server has to be willing to follow a symbolic link for this to work.  Nginx and lighttpd do so by default.  Apache needs `FollowSymLinks` for the directory your files go to, which you can set with an `.htaccess` file in your upload directory:

```apache
# Let the monitor publish data/current.png as a link to the day's chart.
Options +FollowSymLinks
```

Be aware that many shared hosts refuse to let an `.htaccess` file set `Options` at all.  When that happens Apache does not ignore the line, it returns a 500 error for the whole directory, and the server's error log says "Options not allowed here".  If you see that, delete the `.htaccess` file and go back to Copy.

If the page shows a broken image after you switch to Symlink, the server is not following the link.  Switch back to Copy and nothing else needs changing.

Lastly, restart your monitor if it was already running.

## What the page does once it is up

The page updates itself once per minute.  It replaces the chart in place rather than reloading, so it does not flicker.  The "Last data update" line shows the time in your own timezone, wherever you are reading from, taken from the moment the chart reached the server.

At midnight in your station's timezone the monitor starts a new day's chart, which begins with a single data point.  Rather than replace a full day's chart with an empty one, the page stops updating and says "Updating paused.  Refresh this page to resume."  Refresh whenever you're ready to watch the new day.
