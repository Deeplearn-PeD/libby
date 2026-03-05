# SSH Keys Directory

This directory contains SSH keys for the SFTP server authentication.

## Setup

Run the setup script from the project root:

```bash
./scripts/setup-ssh-keys.sh
```

This will generate:
- `ssh_host_ed25519_key` - Private key (keep secure!)
- `ssh_host_ed25519_key.pub` - Public key
- `authorized_keys` - Authorized keys for SFTP user

## Connecting to SFTP

```bash
sftp -i ssh_keys/ssh_host_ed25519_key -P 2222 libby@localhost
```

## Security

- Never commit the private key to version control
- The `ssh_keys/` directory is in `.gitignore`
- Restrict permissions: `chmod 600 ssh_keys/*`
